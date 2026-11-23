package kr.ac.signai.offline.core

data class Sample(
    val timeMs: Long, val sequence: Long, val fsr: DoubleArray, val imu: DoubleArray,
    val imuOk: Boolean, val receivedUtcMs: Long = 0, val receivedElapsedMs: Long = 0,
    val temperature: Double = Double.NaN,
)
enum class Hand(val korean: String) { LEFT("왼손"), RIGHT("오른손") }
data class Capture(val left: List<Sample>, val right: List<Sample>, val replay: Boolean = false)

/** Per-connection fragment assembler. uint32 sequence gaps count once, including expired fragments. */
class PacketDecoder {
    private data class Parts(val started: Long, val chunks: MutableMap<Int, ByteArray> = mutableMapOf())
    private val pending = linkedMapOf<Long, Parts>()
    private val countedLost = linkedSetOf<Long>()
    private var first: Long? = null
    private var last: Long? = null
    var missing: Long = 0; private set
    var samples: Long = 0; private set
    private fun distance(a: Long, b: Long) = (a - b) and 0xffffffffL
    private fun lose(seq: Long) { if (countedLost.add(seq)) missing++ }
    fun reset() { pending.clear(); countedLost.clear(); first = null; last = null; missing = 0; samples = 0 }
    fun accept(packet: ByteArray, utc: Long, elapsed: Long): Sample? {
        expire(elapsed)
        if (packet.size != 20) return null
        val piece = (packet[0].toInt() and 255) - 0xa0
        if (piece !in 0..2) return null
        val seq = java.nio.ByteBuffer.wrap(packet, 1, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xffffffffL
        val previous = last
        if (previous != null && (distance(seq, previous) == 0L || distance(seq, previous) >= 0x80000000L)) return null
        if (seq in countedLost) return null
        if (first == null) first = seq
        val parts = pending.getOrPut(seq) { Parts(elapsed) }
        val bodyPiece = packet.copyOfRange(5, 20)
        if (parts.chunks[piece]?.contentEquals(bodyPiece) == false) {
            pending.remove(seq); lose(seq); return null
        }
        parts.chunks[piece] = bodyPiece
        while (pending.size > 8) { val oldest = pending.keys.first(); pending.remove(oldest); lose(oldest) }
        if (parts.chunks.size != 3) return null
        val body = parts.chunks[0]!! + parts.chunks[1]!! + parts.chunks[2]!!
        val lower = previous ?: ((first!! - 1) and 0xffffffffL)
        val gap = distance(seq, lower) - 1
        if (gap in 1..100000) {
            val already = countedLost.count { distance(it, lower) in 1..gap }
            missing += gap - already
        } else if (gap > 100000) { pending.clear(); countedLost.clear() }
        pending.keys.filter { distance(seq, it) < 0x80000000L }.toList().forEach { pending.remove(it) }
        countedLost.removeAll { distance(seq, it) < 0x80000000L }
        last = seq
        val b = java.nio.ByteBuffer.wrap(body).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val time = b.int.toLong() and 0xffffffffL
        val fsr = DoubleArray(6) { (b.short.toInt() and 65535).toDouble() }
        val imu = DoubleArray(6) { b.float.toDouble() }
        val temp = b.float.toDouble()
        val ok = b.get().toInt() == 1
        samples++
        return Sample(time, seq, fsr, imu, ok, utc, elapsed, temp)
    }
    fun expire(elapsed: Long) {
        pending.filterValues { elapsed - it.started > 500 }.keys.toList().forEach { pending.remove(it); lose(it) }
        while (countedLost.size > 512) countedLost.remove(countedLost.first())
    }
}
