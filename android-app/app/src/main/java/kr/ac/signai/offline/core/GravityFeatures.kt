package kr.ac.signai.offline.core

import kotlin.math.*

/** Exact feature_version=1 contract of sign_gravity_model.py, using doubles. */
object GravityFeatures {
    fun validate(samples: List<Sample>, hand: String) {
        require(samples.size >= 30) { "$hand 샘플 부족: ${samples.size}개 / 최소 30개" }
        require(samples.all { it.fsr.size == 6 && it.imu.size == 6 && it.fsr.all(Double::isFinite) && it.imu.all(Double::isFinite) }) { "$hand 센서에 유효하지 않은 값이 있습니다" }
        val gaps = samples.zipWithNext { a, b -> b.timeMs - a.timeMs }
        require(gaps.all { it > 0 }) { "$hand 보드 시계가 증가하지 않습니다" }
        require(samples.last().timeMs - samples.first().timeMs >= 3000) { "$hand 측정 길이 3초 미만" }
        require(gaps.max() <= 500) { "$hand 수신 간격이 500ms를 넘었습니다" }
        require(samples.count { !it.imuOk }.toDouble() / samples.size <= .05) { "$hand IMU 오류 비율 5% 초과" }
    }
    fun capture(c: Capture): DoubleArray {
        validate(c.left, "왼손"); validate(c.right, "오른손")
        return hand(c.left.map { it.imu }) + hand(c.right.map { it.imu })
    }
    fun percentile(values: DoubleArray, p: Double): Double {
        val sorted = values.sorted(); val index = (sorted.size - 1) * p / 100
        val low = floor(index).toInt(); val high = ceil(index).toInt()
        return sorted[low] + (sorted[high] - sorted[low]) * (index - low)
    }
    private fun norm(x: DoubleArray) = sqrt(x.sumOf { it * it })
    private fun stats(rows: List<DoubleArray>): DoubleArray {
        val mean = DoubleArray(6) { c -> rows.sumOf { it[c] } / rows.size }
        return mean + DoubleArray(6) { c -> sqrt(rows.sumOf { (it[c] - mean[c]).pow(2) } / rows.size) } +
            DoubleArray(6) { c -> rows.minOf { it[c] } } + DoubleArray(6) { c -> rows.maxOf { it[c] } }
    }
    private fun phased(rows: List<DoubleArray>): DoubleArray {
        var result = stats(rows); var offset = 0
        for (i in 0..2) {
            val count = rows.size / 3 + if (i < rows.size % 3) 1 else 0
            result += stats(rows.subList(offset, offset + count)); offset += count
        }
        return result
    }
    fun rotation(g: DoubleArray): Array<DoubleArray> {
        val magnitude = norm(g)
        require(magnitude.isFinite() && magnitude >= 1e-12) { "중력 기준을 계산할 수 없습니다" }
        val a = g.map { it / magnitude }; val cross = doubleArrayOf(a[1], -a[0], 0.0)
        val sine = norm(cross); val cosine = a[2].coerceIn(-1.0, 1.0)
        if (sine < 1e-12) return Array(3) { r -> DoubleArray(3) { c -> if (r != c) 0.0 else if (cosine >= 0 || r == 0) 1.0 else -1.0 } }
        val u = cross.map { it / sine }
        val k = arrayOf(doubleArrayOf(0.0,-u[2],u[1]),doubleArrayOf(u[2],0.0,-u[0]),doubleArrayOf(-u[1],u[0],0.0))
        return Array(3) { r -> DoubleArray(3) { c -> (if (r == c) 1.0 else 0.0) + sine * k[r][c] +
            (1 - cosine) * (0..2).sumOf { j -> k[r][j] * k[j][c] } } }
    }
    fun hand(rows: List<DoubleArray>): DoubleArray {
        require(rows.size >= 8 && rows.all { it.size == 6 && it.all(Double::isFinite) })
        val gm = rows.map { norm(it.copyOfRange(3,6)) }.toDoubleArray()
        val quiet = rows.indices.filter { gm[it] <= percentile(gm,30.0) }
        val g0 = DoubleArray(3) { c -> quiet.sumOf { rows[it][c] } / quiet.size }
        val rotation = rotation(g0)
        fun rotate(v: DoubleArray, offset: Int) = DoubleArray(3) { r -> (0..2).sumOf { c -> rotation[r][c] * v[c + offset] } }
        val gf = rows.mapIndexed { i, v ->
            val a = rotate(v,0); val w = rotate(v,3)
            doubleArrayOf(a[2],hypot(a[0],a[1]),w[2],hypot(w[0],w[1]),norm(DoubleArray(3) { v[it]-g0[it] }),gm[i])
        }
        val threshold = max(10.0,percentile(gm,20.0)*2+5)
        val active = rows.indices.filter { gm[it] > threshold }
        var s = 0; var e = rows.size
        if (active.size >= 2) {
            s = max(0,active.first()-2); e = min(rows.size,active.last()+3)
            if (e-s < 8) { s = 0; e = rows.size }
        }
        val window = gf.subList(s,e)
        val resampled = DoubleArray(72) { i ->
            val index = (i/6).toDouble()/11 * (window.size-1)
            val low = floor(index).toInt(); val high = ceil(index).toInt().coerceAtMost(window.lastIndex)
            window[low][i%6] + (window[high][i%6]-window[low][i%6]) * (index-low)
        }
        return (phased(gf)+phased(window)+doubleArrayOf((e-s).toDouble()/rows.size,gm.count { it>30 }.toDouble()/rows.size)+resampled)
            .also { require(it.size == 266 && it.all(Double::isFinite)) }
    }
}
