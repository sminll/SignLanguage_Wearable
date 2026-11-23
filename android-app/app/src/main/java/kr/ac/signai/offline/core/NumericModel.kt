package kr.ac.signai.offline.core

import org.json.JSONObject
import kotlin.math.exp

data class Prediction(val sign: String, val probability: Double, val probabilities: List<Double>, val accepted: Boolean)
class NumericModel private constructor(
    val signs: List<String>, private val mean: DoubleArray, private val scale: DoubleArray,
    private val coef: List<DoubleArray>, private val intercept: DoubleArray,
    val sourceRef: String, val sourceCommit: String, val trainedTrials: Int, val created: String, val note: String,
) {
    fun predict(x: DoubleArray, threshold: Double): Prediction {
        require(x.size == 532 && x.all(Double::isFinite)) { "특징 형식이 다릅니다" }
        require(threshold in 0.0..1.0)
        val z = DoubleArray(532) { (x[it]-mean[it])/scale[it] }
        val logits = coef.indices.map { r -> intercept[r]+z.indices.sumOf { c -> coef[r][c]*z[c] } }
        val probabilities = if (signs.size == 2) {
            val v = logits.single(); val p = if (v >= 0) 1/(1+exp(-v)) else exp(v)/(1+exp(v))
            listOf(1-p,p)
        } else {
            val largest = logits.max(); val powers = logits.map { exp(it-largest) }; val sum = powers.sum()
            powers.map { it/sum }
        }
        require(probabilities.all(Double::isFinite)) { "모델 계산 결과가 유효하지 않습니다" }
        val best = probabilities.indices.maxBy { probabilities[it] }
        return Prediction(signs[best],probabilities[best],probabilities,probabilities[best]>=threshold)
    }
    companion object {
        fun parse(text: String): NumericModel {
            val j = JSONObject(text.removePrefix("\uFEFF"))
            require(j.getInt("version")==1 && j.getInt("feature_count")==532) { "지원하지 않는 모델 버전" }
            val array = j.getJSONArray("signs")
            val signs = List(array.length()) { array.getString(it) }
            require(signs.size>=2 && signs.distinct().size==signs.size && signs.all { it.isNotBlank() })
            fun numbers(key: String): DoubleArray {
                val a = j.getJSONArray(key)
                return DoubleArray(a.length()) { a.getDouble(it) }.also { require(it.all(Double::isFinite)) }
            }
            val mean = numbers("mean"); val scale = numbers("scale"); val intercept = numbers("intercept")
            val matrix = j.getJSONArray("coef")
            val coef = List(matrix.length()) { r -> val a=matrix.getJSONArray(r); DoubleArray(a.length()) { a.getDouble(it) } }
            require(mean.size==532 && scale.size==532 && scale.all { it>0 }) { "정규화 모델 형식 오류" }
            require(coef.size==(if(signs.size==2)1 else signs.size) && intercept.size==coef.size &&
                coef.all { it.size==532 && it.all(Double::isFinite) }) { "분류 모델 형식 오류" }
            return NumericModel(signs,mean,scale,coef,intercept,j.getString("source_ref"),j.getString("source_commit"),
                j.getInt("trained_trials"),j.getString("created"),j.getString("note"))
        }
    }
}
