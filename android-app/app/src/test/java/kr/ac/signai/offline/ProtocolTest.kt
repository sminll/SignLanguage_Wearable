package kr.ac.signai.offline

import kr.ac.signai.offline.core.*
import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ProtocolTest {
    /** Same explicit field offsets and fragment layout as firmware encodePackets. */
    private fun encode(seq:Long=42,time:Long=5000,valid:Boolean=true):List<ByteArray> {
        val body=ByteBuffer.allocate(45).order(ByteOrder.LITTLE_ENDIAN)
        body.putInt(time.toInt());(1..6).forEach { body.putShort((it*100).toShort()) }
        listOf(.125f,-.5f,1f,12f,-20f,3f,25.5f).forEach { body.putFloat(if(valid)it else Float.NaN) };body.put(if(valid)1.toByte() else 0.toByte())
        return (0..2).map { i -> ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN).put((0xa0+i).toByte()).putInt(seq.toInt()).put(body.array(),i*15,15).array() }
    }
    private fun feed(decoder:PacketDecoder,packets:List<ByteArray>,elapsed:Long=1000):Sample? = packets.map { decoder.accept(it,1700000000000,elapsed) }.lastOrNull { it!=null }
    @Test fun firmwareLayoutAndShuffledPieces() {
        val d=PacketDecoder();val p=encode();val sample=feed(d,listOf(p[2],p[0],p[1]))!!
        assertEquals(42L,sample.sequence);assertEquals(5000L,sample.timeMs)
        assertArrayEquals(doubleArrayOf(100.0,200.0,300.0,400.0,500.0,600.0),sample.fsr,0.0)
        assertArrayEquals(doubleArrayOf(.125,-.5,1.0,12.0,-20.0,3.0),sample.imu,0.0)
        assertEquals(25.5,sample.temperature,0.0);assertTrue(sample.imuOk)
    }
    @Test fun invalidImuKeepsFirmwareNaNs() { val sample=feed(PacketDecoder(),encode(valid=false))!!;assertFalse(sample.imuOk);assertTrue(sample.imu.all { it.isNaN() });assertTrue(sample.temperature.isNaN()) }
    @Test fun partialSequenceCountsOnceAndCannotBeMixed() {
        val d=PacketDecoder();d.accept(encode(0)[0],0,0)
        val sample=feed(d,encode(1),100)!!
        assertEquals(1L,sample.sequence);assertEquals(1L,d.missing)
        assertNull(feed(d,encode(0),200));assertEquals(1L,d.samples)
    }
    @Test fun timeoutAndGapDoNotDoubleCount() {
        val d=PacketDecoder();d.accept(encode(0)[0],0,0);d.expire(600)
        assertEquals(1L,d.missing);feed(d,encode(1),700);assertEquals(1L,d.missing)
    }
    @Test fun wholeMissingSequenceAndUint32Wrap() {
        val d=PacketDecoder();feed(d,encode(0xfffffffeL));feed(d,encode(0L),1100)
        assertEquals(1L,d.missing);assertEquals(2L,d.samples)
    }
    @Test fun twoBoardsNeverShareFragments() {
        val left=PacketDecoder();val right=PacketDecoder();val p=encode()
        left.accept(p[0],0,0);right.accept(p[1],0,0);right.accept(p[2],0,0)
        assertEquals(0L,left.samples);assertEquals(0L,right.samples)
    }
    @Test fun malformedAndDuplicatePacketsDoNotEmit() {
        val d=PacketDecoder();assertNull(d.accept(ByteArray(19),0,0));val p=encode()
        feed(d,p);assertNull(feed(d,p));assertEquals(1L,d.samples)
    }
    private fun samples()=(0 until 50).map { Sample(it*100L,it.toLong(),DoubleArray(6),doubleArrayOf(0.0,0.0,1.0,0.0,0.0,0.0),true) }
    @Test fun qcRejectsMissingHandAndFiniteClockGaps() {
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.capture(Capture(samples(),emptyList())) }
        val bad=samples().mapIndexed { i,s -> if(i>=25)s.copy(timeMs=s.timeMs+1000) else s }
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.validate(bad,"test") }
    }
    @Test fun qcRejectsShortClockAndNonfinite() {
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.validate(samples().take(29),"test") }
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.validate(samples().map { it.copy(timeMs=1) },"test") }
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.validate(samples().map { it.copy(imu=DoubleArray(6) { Double.NaN }) },"test") }
    }
    @Test fun qcAllowsAtMostFivePercentInvalidFlags() {
        GravityFeatures.validate(samples().mapIndexed { i,s -> s.copy(imuOk=i>=2) },"test")
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.validate(samples().mapIndexed { i,s -> s.copy(imuOk=i>=3) },"test") }
    }
    @Test fun zeroGravityRefusedAndNegativeZHandled() {
        assertThrows(IllegalArgumentException::class.java) { GravityFeatures.hand(List(50) { DoubleArray(6) }) }
        val result=GravityFeatures.hand(List(50) { doubleArrayOf(0.0,0.0,-1.0,0.0,0.0,0.0) })
        assertEquals(266,result.size);assertTrue(result.all(Double::isFinite))
    }
}
