package kr.ac.signai.offline

import android.app.Application
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kr.ac.signai.offline.ble.*
import kr.ac.signai.offline.core.*

data class Word(val id:Long,val text:String,val replay:Boolean)
data class UiState(val devices:List<BandDevice> = emptyList(),val bluetooth:Boolean=false,val scanning:Boolean=false,
    val message:String="",val stage:String="",val busy:Boolean=false,val result:Prediction?=null,val replay:Boolean=false,
    val words:List<Word> = emptyList(),val threshold:Double=.5,val left:List<Sample> = emptyList(),val right:List<Sample> = emptyList())

class DemoViewModel(application:Application):AndroidViewModel(application) {
    private val preferences=application.getSharedPreferences("settings",0)
    private val mutable=MutableStateFlow(UiState(threshold=preferences.getFloat("threshold",.5f).toDouble()))
    val state=mutable.asStateFlow()
    val model=runCatching { NumericModel.parse(application.assets.open("sign_model.json").bufferedReader().use { it.readText() }) }.getOrNull()
    private var manager:BandManager?=null
    val bands:BandManager get()=manager!!
    private var job:Job?=null
    private var start:Long?=null
    private val recorded=Hand.entries.associateWith { mutableListOf<Sample>() }
    private var wordId=0L
    init {
        manager=BandManager(application,{ updateBands() },{ hand,sample->sample(hand,sample) })
        if(model==null)message("모델을 읽을 수 없습니다. 앱을 다시 설치하세요")
    }
    private fun updateBands() {
        val b=manager?:return
        mutable.value=mutable.value.copy(devices=b.devices.values.toList(),bluetooth=b.available(),scanning=b.scanning,message=b.message.ifBlank { mutable.value.message })
        if(start!=null && !b.bothReady())cancel("측정 취소: 밴드 연결이 끊겼습니다")
    }
    private fun sample(hand:Hand,s:Sample) {
        val now=s.receivedElapsedMs
        val values=(if(hand==Hand.LEFT)mutable.value.left else mutable.value.right).filter { now-it.receivedElapsedMs<=10000 }.takeLast(199)+s
        mutable.value=if(hand==Hand.LEFT)mutable.value.copy(left=values) else mutable.value.copy(right=values)
        start?.let { if(now>=it && now<it+5000)recorded.getValue(hand).add(s) }
    }
    fun message(text:String) { mutable.value=mutable.value.copy(message=text) }
    fun threshold(value:Double) { val bounded=value.coerceIn(.0,1.0);preferences.edit().putFloat("threshold",bounded.toFloat()).apply();mutable.value=mutable.value.copy(threshold=bounded) }
    fun clear() { mutable.value=mutable.value.copy(words=emptyList()) }
    fun delete(id:Long) { mutable.value=mutable.value.copy(words=mutable.value.words.filterNot { it.id==id }) }
    fun sentence()=mutable.value.words.joinToString(" ") { it.text }
    fun measure() {
        if(mutable.value.busy)return
        if(model==null) { message("모델이 없습니다");return }
        if(!bands.fresh()) { message("양손 밴드의 최신 데이터가 필요합니다. 연결·센서 수신 상태를 확인하세요");return }
        job=viewModelScope.launch {
            mutable.value=mutable.value.copy(busy=true,result=null,replay=false,message="")
            try {
                for(n in 3 downTo 1) { mutable.value=mutable.value.copy(stage="$n");delay(1000) }
                require(bands.fresh()) { "양손 센서 수신이 끊겼습니다" }
                recorded.values.forEach { it.clear() };start=SystemClock.elapsedRealtime()
                for(n in 5 downTo 1) { mutable.value=mutable.value.copy(stage="측정 중 · ${n}초");delay(1000) }
                start=null
                val capture=Capture(recorded.getValue(Hand.LEFT).toList(),recorded.getValue(Hand.RIGHT).toList())
                infer(capture)
            } catch(e:CancellationException) { throw e }
            catch(e:Exception) { message("판정하지 않았습니다: ${e.message}") }
            finally { if(job===currentCoroutineContext()[Job]) { start=null;mutable.value=mutable.value.copy(busy=false,stage="") } }
        }
    }
    private suspend fun infer(capture:Capture) {
        val threshold=mutable.value.threshold
        val prediction=withContext(Dispatchers.Default) { model!!.predict(GravityFeatures.capture(capture),threshold) }
        currentCoroutineContext().ensureActive()
        val words=if(prediction.accepted)mutable.value.words+Word(++wordId,prediction.sign,capture.replay) else mutable.value.words
        mutable.value=mutable.value.copy(result=prediction,replay=capture.replay,words=words,
            message=if(prediction.accepted)"" else "다시 해 주세요 · 설정한 판정 임계값보다 낮습니다")
    }
    fun replay(text:suspend ()->String) {
        if(mutable.value.busy)return
        if(model==null) { message("모델이 없습니다");return }
        job=viewModelScope.launch {
            mutable.value=mutable.value.copy(busy=true,stage="CSV 재생 중",result=null,message="",replay=true)
            try { val capture=withContext(Dispatchers.IO) { CsvReplay.parse(text()) };infer(capture) }
            catch(e:CancellationException) { throw e }
            catch(e:Exception) { message("CSV 판정 실패: ${e.message}") }
            finally { if(job===currentCoroutineContext()[Job])mutable.value=mutable.value.copy(busy=false,stage="") }
        }
    }
    fun cancel(reason:String="측정을 취소했습니다") { job?.cancel();job=null;start=null;recorded.values.forEach { it.clear() };mutable.value=mutable.value.copy(busy=false,stage="",message=reason) }
    fun background() { if(mutable.value.busy)cancel("백그라운드 전환으로 측정을 취소했습니다");bands.setForeground(false) }
    fun foreground() { bands.setForeground(true);updateBands() }
    override fun onCleared() { bands.close();super.onCleared() }
}
