package kr.ac.signai.offline

import kr.ac.signai.offline.core.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class ParityTest {
    private fun resource(name:String)=javaClass.classLoader!!.getResourceAsStream(name)!!.bufferedReader().use { it.readText() }
    private fun vectors()=JSONObject(resource("parity_vectors.json").removePrefix("\uFEFF")).getJSONArray("vectors")
    @Test fun fiveRealCsvsMatch532PythonFeaturesAndProbabilities() {
        val model=NumericModel.parse(resource("sign_model.json"));val fixtures=vectors()
        assertEquals(725,model.trainedTrials)
        assertEquals(setOf("삼키다","아니다","상처"),model.signs.toSet())
        assertEquals(5,fixtures.length())
        var featureError=0.0;var probabilityError=0.0
        for(i in 0 until fixtures.length()) {
            val v=fixtures.getJSONObject(i);val text=resource(v.getString("csv"))
            val features=GravityFeatures.capture(CsvReplay.parse(text))
            assertEquals(532,features.size)
            val expected=v.getJSONArray("features")
            features.forEachIndexed { c,value -> featureError=max(featureError,abs(value-expected.getDouble(c)));assertEquals(expected.getDouble(c),value,1e-6) }
            val result=model.predict(features,.5)
            val probabilities=v.getJSONArray("probabilities")
            result.probabilities.forEachIndexed { c,value -> probabilityError=max(probabilityError,abs(value-probabilities.getDouble(c)));assertEquals(probabilities.getDouble(c),value,1e-6) }
            assertEquals(v.getString("prediction"),result.sign)
        }
        println("Real CSV parity: max feature error=$featureError; max probability error=$probabilityError")
    }
    @Test fun actualCaptureFeaturesInvariantUnderCommonThreeDimensionalRotation() {
        val fixture=vectors().getJSONObject(0)
        val original=CsvReplay.parse(resource(fixture.getString("csv")))
        val a=.61;val b=-.87;val c=1.18
        val rx=arrayOf(doubleArrayOf(1.0,0.0,0.0),doubleArrayOf(0.0,cos(a),-sin(a)),doubleArrayOf(0.0,sin(a),cos(a)))
        val ry=arrayOf(doubleArrayOf(cos(b),0.0,sin(b)),doubleArrayOf(0.0,1.0,0.0),doubleArrayOf(-sin(b),0.0,cos(b)))
        val rz=arrayOf(doubleArrayOf(cos(c),-sin(c),0.0),doubleArrayOf(sin(c),cos(c),0.0),doubleArrayOf(0.0,0.0,1.0))
        fun rotate(matrix:Array<DoubleArray>,v:DoubleArray)=DoubleArray(3) { r -> (0..2).sumOf { col -> matrix[r][col]*v[col] } }
        fun group(samples:List<Sample>)=samples.map { sample ->
            fun transform(offset:Int)=rotate(rz,rotate(ry,rotate(rx,sample.imu.copyOfRange(offset,offset+3))))
            sample.copy(imu=transform(0)+transform(3))
        }
        assertArrayEquals(GravityFeatures.capture(original),GravityFeatures.capture(Capture(group(original.left),group(original.right))),1e-6)
        assertArrayEquals(doubleArrayOf(0.0,0.0,1.0),GravityFeatures.rotation(doubleArrayOf(0.0,0.0,-1.0)).map { it[2]*-1 }.toDoubleArray(),1e-12)
    }
    @Test fun fsrMagnitudeDoesNotEnterImuOnlyFeatures() {
        val capture=CsvReplay.parse(resource(vectors().getJSONObject(0).getString("csv")))
        val altered=Capture(capture.left.map { it.copy(fsr=DoubleArray(6) { 4095.0 }) },capture.right.map { it.copy(fsr=DoubleArray(6) { 0.0 }) })
        assertArrayEquals(GravityFeatures.capture(capture),GravityFeatures.capture(altered),0.0)
    }
    @Test fun thresholdDoesNotChangeClassAndRejectsLowOutput() {
        val model=NumericModel.parse(resource("sign_model.json"))
        val x=GravityFeatures.capture(CsvReplay.parse(resource(vectors().getJSONObject(0).getString("csv"))))
        val a=model.predict(x,0.0);val b=model.predict(x,1.0)
        assertTrue(a.accepted);assertFalse(b.accepted);assertEquals(a.sign,b.sign)
    }
    @Test fun binaryOneRowSklearnRule() {
        val j=JSONObject(resource("sign_model.json").removePrefix("\uFEFF"))
        j.put("signs",org.json.JSONArray(listOf("a","b")))
        j.put("mean",org.json.JSONArray(List(532) { 0.0 }));j.put("scale",org.json.JSONArray(List(532) { 1.0 }))
        j.put("coef",org.json.JSONArray(listOf(List(532) { 0.0 })));j.put("intercept",org.json.JSONArray(listOf(2.0)))
        val result=NumericModel.parse(j.toString()).predict(DoubleArray(532),.5)
        assertEquals("b",result.sign);assertEquals(1/(1+exp(-2.0)),result.probability,1e-12)
    }
    @Test fun malformedModelIsRejected() {
        val j=JSONObject(resource("sign_model.json").removePrefix("\uFEFF"));j.put("feature_count",12)
        assertThrows(IllegalArgumentException::class.java) { NumericModel.parse(j.toString()) }
    }
    @Test fun csvQuotingAndTrueLabelsDoNotRouteInference() {
        val text=resource(vectors().getJSONObject(0).getString("csv"))
        val records=CsvReplay.records(text);val index=records.first().indexOf("label")
        val changed=records.mapIndexed { r,row -> row.mapIndexed { col,value ->
            val s=if(r>0&&col==index)"unknown,label" else value
            "\""+s.replace("\"","\"\"")+"\""
        }.joinToString(",") }.joinToString("\r\n")
        assertArrayEquals(GravityFeatures.capture(CsvReplay.parse(text)),GravityFeatures.capture(CsvReplay.parse(changed)),0.0)
    }
    @Test fun numpyLinearPercentile() { assertEquals(2.7,GravityFeatures.percentile(doubleArrayOf(0.0,3.0,9.0,10.0),30.0),1e-12) }
}
