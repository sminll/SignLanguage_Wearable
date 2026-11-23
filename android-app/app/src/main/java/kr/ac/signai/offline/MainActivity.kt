package kr.ac.signai.offline

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import kr.ac.signai.offline.core.*
import java.util.Locale
import kotlin.math.*

private val Blue=Color(0xFF6C9FFF)
private val Yellow=Color(0xFFFFD36A)
private val Panel=Color(0xFF171D2A)
private val Subtle=Color(0xFF9BA8BC)
private val Background=Color(0xFF0B101A)

class MainActivity:ComponentActivity() {
    private val vm:DemoViewModel by viewModels()
    private var tts:TextToSpeech?=null
    private var offlineVoice=false
    private val permissions=registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
        if(result.values.all { it })scanAllowed() else vm.message("밴드 검색에는 근처 기기 권한이 필요합니다. 아래 앱 권한 설정에서 허용해 주세요")
    }
    private val chooser=registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if(uri!=null)vm.replay { readLimited(uri) }
    }
    override fun onCreate(savedInstanceState:Bundle?) {
        super.onCreate(savedInstanceState)
        tts=TextToSpeech(this) { status ->
            if(status==TextToSpeech.SUCCESS) {
                val voice=tts?.voices?.firstOrNull { it.locale.language=="ko" && !it.isNetworkConnectionRequired }
                if(voice!=null) { tts?.voice=voice;offlineVoice=true }
            }
        }
        setContent {
            MaterialTheme(colorScheme=darkColorScheme(primary=Blue,background=Background,surface=Panel,onSurface=Color(0xFFF1F5FF),secondary=Yellow)) {
                val state by vm.state.collectAsStateWithLifecycle()
                SignApp(vm,state,::search,::permissionSettings,::bluetoothSettings,::speak,
                    { chooser.launch(arrayOf("text/csv","text/comma-separated-values","text/plain","application/octet-stream")) },
                    { n -> vm.replay { assets.open("replay/parity_$n.csv").bufferedReader().use { it.readText() } } })
            }
        }
    }
    private fun readLimited(uri:Uri):String {
        val stream=contentResolver.openInputStream(uri)?:error("파일을 열 수 없습니다")
        return stream.bufferedReader(Charsets.UTF_8).use { reader ->
            val output=StringBuilder();val buffer=CharArray(4096)
            while(true) { val n=reader.read(buffer);if(n<0)break;require(output.length+n<=2_000_000) { "CSV가 너무 큽니다 (최대 2MB)" };output.append(buffer,0,n) }
            output.toString()
        }
    }
    private fun search() {
        val needed=if(Build.VERSION.SDK_INT>=31)arrayOf(Manifest.permission.BLUETOOTH_SCAN,Manifest.permission.BLUETOOTH_CONNECT)
            else arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
        if(needed.any { checkSelfPermission(it)!=PackageManager.PERMISSION_GRANTED })permissions.launch(needed) else scanAllowed()
    }
    private fun scanAllowed() {
        if(Build.VERSION.SDK_INT<=30 && !(getSystemService(LOCATION_SERVICE) as LocationManager).isLocationEnabled) {
            vm.message("Android 11 이하는 BLE 검색에 위치 기능이 필요합니다. 휴대폰 위치 기능을 켜 주세요");return
        }
        vm.bands.scan()
    }
    private fun permissionSettings() { startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,Uri.parse("package:$packageName"))) }
    private fun bluetoothSettings() { startActivity(Intent(Settings.ACTION_BLUETOOTH_SETTINGS)) }
    private fun speak() {
        val sentence=vm.sentence()
        if(sentence.isBlank()) { vm.message("먼저 수어를 판정해 주세요");return }
        if(!offlineVoice) { vm.message("오프라인 한국어 음성이 필요합니다. 휴대폰의 글자 읽어주기(TTS) 설정에서 한국어 음성 데이터를 설치하세요");return }
        tts?.speak(sentence,TextToSpeech.QUEUE_FLUSH,null,"sign-sentence")
    }
    override fun onStart() { super.onStart();vm.foreground() }
    override fun onStop() { vm.background();tts?.stop();super.onStop() }
    override fun onDestroy() { tts?.shutdown();super.onDestroy() }
}

@Composable
private fun SignApp(vm:DemoViewModel,state:UiState,search:()->Unit,permissions:()->Unit,bluetooth:()->Unit,speak:()->Unit,pickCsv:()->Unit,example:(Int)->Unit) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    val tabs=listOf("동작 인식","보정","센서","학습","설정")
    Scaffold(containerColor=Background,bottomBar={
        Card(Modifier.navigationBarsPadding().padding(horizontal=16.dp,vertical=10.dp).fillMaxWidth(),shape=RoundedCornerShape(28.dp),colors=CardDefaults.cardColors(containerColor=Panel)) {
            Row(Modifier.padding(vertical=10.dp)) { tabs.forEachIndexed { i,label ->
                Column(Modifier.weight(1f).clickable { tab=i }.padding(vertical=4.dp),horizontalAlignment=Alignment.CenterHorizontally) {
                    NavIcon(i,tab==i);Text(label,fontSize=10.sp,color=if(tab==i)Blue else Subtle)
                }
            } }
        }
    }) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal=20.dp),verticalArrangement=Arrangement.spacedBy(16.dp)) {
            Spacer(Modifier.height(8.dp))
            Text("SIGN:AI",color=Blue,fontSize=14.sp,fontWeight=FontWeight.Bold)
            Text(when(tab) { 0->"수어 번역";1->"밴드 보정";2->"실시간 센서";3->"학습 데이터";else->"설정" },fontSize=30.sp,fontWeight=FontWeight.Bold)
            if(state.message.isNotBlank())Text(state.message,color=Yellow,modifier=Modifier.fillMaxWidth().background(Panel,RoundedCornerShape(12.dp)).padding(14.dp))
            when(tab) {
                0 -> {
                    Tile("나의 문장") {
                        if(state.words.isEmpty())Text("수어를 하면 여기에 표시됩니다",color=Subtle)
                        else Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                            Text(state.words.joinToString(" ") { it.text },color=Blue,fontSize=28.sp,fontWeight=FontWeight.Bold)
                            Row(Modifier.horizontalScroll(rememberScrollState()),horizontalArrangement=Arrangement.spacedBy(8.dp)) {
                                state.words.forEach { word -> Text(word.text+(if(word.replay)" · 재생" else ""),color=Blue,
                                    modifier=Modifier.background(Blue.copy(alpha=.12f),RoundedCornerShape(12.dp)).pointerInput(word.id) {
                                        detectTapGestures(onLongPress={vm.delete(word.id)},onTap={vm.message("개별 수어 영상은 담당자 확인 후 제공됩니다")})
                                    }.padding(10.dp)) }
                            }
                            Text("단어를 길게 누르면 삭제됩니다",color=Subtle,fontSize=11.sp)
                        }
                        Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick=vm::clear,enabled=state.words.isNotEmpty()) { Text("전체 지우기") }
                            Button(onClick=speak,enabled=state.words.isNotEmpty()) { Text("말하기") }
                        }
                    }
                    state.result?.let { result -> Tile(if(state.replay)"재생 데이터 · 판정 결과" else "판정 결과") {
                        Text(if(result.accepted)result.sign else "다시 해 주세요",fontSize=30.sp,color=if(result.accepted)Blue else Yellow,fontWeight=FontWeight.Bold)
                        Text("모델 출력 ${(result.probability*100).format(1)}% · 임계값 ${(state.threshold*100).format(0)}%",color=Subtle)
                        if(state.replay)Text("CSV 재생 결과입니다. 실시간 인식 성능을 측정한 결과가 아닙니다",color=Yellow,fontSize=12.sp)
                    } }
                    Button(onClick=vm::measure,enabled=!state.busy&&vm.model!=null,modifier=Modifier.fillMaxWidth().height(64.dp),shape=RoundedCornerShape(20.dp)) {
                        Text(if(state.busy)state.stage else "측정",fontSize=21.sp,fontWeight=FontWeight.Bold)
                    }
                    if(state.busy)OutlinedButton(onClick={vm.cancel()},modifier=Modifier.fillMaxWidth()) { Text("취소") }
                    Text("시작 직후 손을 잠깐 멈췄다가 수어를 하고, 끝나면 다시 멈추세요.\n3·2·1 준비 후 5초 동안 양손을 측정합니다.",color=Subtle,fontSize=13.sp)
                    Tile("문장 수어 영상 · 준비 중") {
                        Button(onClick={},enabled=false,modifier=Modifier.fillMaxWidth()) { Text("문장 수어 영상 재생") }
                        Row(verticalAlignment=Alignment.CenterVertically) { Checkbox(false,onCheckedChange=null,enabled=false);Text("영상 확인 후 말하기 · 준비 중",color=Subtle) }
                        Text("단어를 누르면 개별 영상을 확인할 수 있어요. 문장 영상을 모두 확인한 뒤 ‘맞아요’를 누르면 문장을 말합니다.",color=Subtle,fontSize=12.sp)
                        Text("영상 출처와 재생 규칙을 iOS 담당자에게 확인 중입니다",color=Yellow,fontSize=12.sp)
                    }
                    BandCard(vm,state,search,permissions,bluetooth)
                    Tile("저장된 CSV로 판정") {
                        Text("실제 밴드 없이 같은 특징·모델 계산을 확인합니다. 포함된 예시는 학습 자료이며 정확도 평가용이 아닙니다.",color=Subtle,fontSize=12.sp)
                        OutlinedButton(onClick=pickCsv,enabled=!state.busy,modifier=Modifier.fillMaxWidth()) { Text("CSV 파일 선택") }
                        Row(Modifier.horizontalScroll(rememberScrollState()),horizontalArrangement=Arrangement.spacedBy(6.dp)) { (1..5).forEach { n ->
                            OutlinedButton(onClick={example(n)},enabled=!state.busy) { Text("예시 $n") }
                        } }
                    }
                }
                1 -> {
                    Tile("선택 보정 · 미검증") {
                        Text("2단계 기능입니다. 1단계 실기기 검증 후 진행합니다.",color=Yellow)
                        OutlinedButton(onClick={},enabled=false) { Text("표준 JSON 가져오기 · 준비 중") }
                    }
                    Hand.entries.forEach { hand -> Tile(hand.korean) {
                        Text(state.devices.firstOrNull { it.hand==hand&&it.connected }?.status?:"연결되지 않음",color=Subtle)
                        listOf("휴식","주먹 쥐기","손바닥 펴기").forEach { label -> Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.SpaceBetween) {
                            Text(label);OutlinedButton(onClick={},enabled=false) { Text("5초 측정") }
                        } }
                    } }
                    Text("휴식 → 주먹 쥐기 → 손바닥 펴기 순서로 측정하세요. 밴드 위치나 조임을 바꾸면 처음부터 다시 측정하세요.",color=Subtle)
                    TextButton(onClick={},enabled=false) { Text("처음부터 다시 보정",color=Color(0xFFEC858D)) }
                }
                2 -> Hand.entries.forEach { hand ->
                    val samples=if(hand==Hand.LEFT)state.left else state.right
                    val devices=state.devices.filter { it.hand==hand }
                    Tile(hand.korean) {
                        SensorGraph("FSR 6개 · ADC",samples,6,true)
                        SensorGraph("가속도 · g",samples,3,false,0)
                        SensorGraph("각속도 · °/s",samples,3,false,3)
                        Text("샘플 ${devices.sumOf { it.samples }} · 누락 프레임 ${devices.sumOf { it.missing }}",color=Subtle,fontSize=12.sp)
                        Text("누락률 "+(devices.sumOf { it.missing }.toDouble()/max(1L,devices.sumOf { it.samples+it.missing })*100).format(2)+"% · 현재 연결 기준",color=Subtle,fontSize=11.sp)
                    }
                }
                3 -> Tile("새 학습 데이터 · 준비 중") {
                    Text("1단계 실기기 검증 후 참가자 선택·20회 측정·CSV 저장·업로드를 진행합니다.",color=Subtle)
                    Text("5초 측정 · 회차 사이 3초 휴식 · 회차별 CSV 저장",color=Subtle)
                    Button(onClick={},enabled=false) { Text("20회 연속 측정") }
                    OutlinedButton(onClick={},enabled=false) { Text("5초 수집하고 CSV 저장") }
                    Text("현재 앱은 측정값을 저장하거나 학습·업로드하지 않습니다.",color=Yellow)
                }
                else -> {
                    Tile("학습 모델 관리") {
                        vm.model?.let { model ->
                            Text(model.signs.joinToString(" · "),color=Blue,fontSize=21.sp,fontWeight=FontWeight.Bold)
                            Text("로지스틱 회귀 · 중력 기준 특징 532개")
                            Text("학습 ${model.trainedTrials}회 · ${model.sourceRef}",color=Subtle)
                            Text(model.sourceCommit,color=Subtle,fontSize=11.sp)
                            Text("생성: ${model.created}",color=Subtle,fontSize=11.sp)
                            Text(model.note,color=Subtle,fontSize=12.sp)
                        }?:Text("모델 없음",color=Yellow)
                        Text("판정 임계값 ${(state.threshold*100).format(0)}%")
                        Slider(value=state.threshold.toFloat(),onValueChange={vm.threshold(it.toDouble())},valueRange=0f..1f)
                        Text("임계값보다 낮으면 문장에 넣지 않습니다. 이 값은 검증된 정답률을 뜻하지 않습니다.",color=Subtle,fontSize=12.sp)
                    }
                    Tile("학습 데이터 업데이트 · 2단계") {
                        Row(verticalAlignment=Alignment.CenterVertically) { Checkbox(false,null,enabled=false);Text("앱 실행 시 자동 업데이트",color=Subtle) }
                        Text("sminll/SignLanguage_Wearable\nmain / android-app/models/sign_model.json",color=Subtle,fontSize=12.sp)
                        Button(onClick={},enabled=false) { Text("지금 업데이트 확인") }
                        Text("자동 업데이트 꺼짐 · 저장된 모델을 사용합니다.",color=Blue)
                    }
                    Tile("GitHub 로그인 · 준비 중") {
                        Text("로그인 계정: 없음\n팀 OAuth App Client ID: 담당자 확인 필요",color=Subtle)
                        Button(onClick={},enabled=false) { Text("GitHub 로그인") }
                        TextButton(onClick={},enabled=false) { Text("로그아웃",color=Color(0xFFEC858D)) }
                        Text("GitHub 공식 화면에서 계정과 비밀번호를 입력합니다. 앱에는 비밀번호를 저장하지 않습니다.",color=Subtle,fontSize=12.sp)
                    }
                    Tile("팀 공개 저장소 · 준비 중") {
                        Text("소유자: sminll\n저장소: SignLanguage_Wearable\n데이터 업로드 브랜치: 미설정",color=Subtle)
                        OutlinedButton(onClick={},enabled=false) { Text("저장소 쓰기 권한 확인") }
                        Text("업로드 결과 · 성공 0 · 실패 0",color=Subtle)
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable private fun Tile(title:String,body:@Composable ColumnScope.()->Unit) {
    Card(Modifier.fillMaxWidth(),shape=RoundedCornerShape(22.dp),colors=CardDefaults.cardColors(containerColor=Panel)) {
        Column(Modifier.padding(18.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) { Text(title,fontSize=16.sp,fontWeight=FontWeight.SemiBold);body() }
    }
}
@Composable private fun BandCard(vm:DemoViewModel,state:UiState,search:()->Unit,permissions:()->Unit,bluetooth:()->Unit) {
    Tile("밴드 연결") {
        Row(Modifier.fillMaxWidth(),verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.SpaceBetween) {
            Text(if(state.bluetooth)"Bluetooth 사용 가능" else "Bluetooth 꺼짐 / 권한 필요",color=Subtle,fontSize=12.sp)
            TextButton(onClick=search) { Text(if(state.scanning)"검색 중…" else "검색") }
        }
        Hand.entries.forEach { hand -> Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) {
            Text("● ${hand.korean}",color=if(hand==Hand.LEFT)Blue else Yellow)
            Text(state.devices.firstOrNull { it.hand==hand&&it.connected }?.status?:"연결되지 않음",fontSize=12.sp,color=Subtle)
        } }
        state.devices.forEach { d ->
            HorizontalDivider(color=Subtle.copy(alpha=.15f))
            Text(d.name,fontWeight=FontWeight.SemiBold)
            Text("${d.address} · ${d.rssi} dBm",fontSize=11.sp,color=Subtle)
            Text(d.status,color=if(d.ready)Blue else Subtle,fontSize=12.sp)
            Row(horizontalArrangement=Arrangement.spacedBy(6.dp),verticalAlignment=Alignment.CenterVertically) {
                Hand.entries.forEach { hand -> FilterChip(selected=d.hand==hand,onClick={vm.bands.assign(d.address,hand)},enabled=!d.connected,label={Text(hand.korean)}) }
                if(d.connected||d.status.contains("연결 중")||d.status.contains("재연결")||d.status.startsWith("연결 끊김"))TextButton(onClick={vm.bands.disconnect(d.address)}) { Text("해제") }
                else Button(onClick={vm.bands.connect(d.address)}) { Text("연결") }
            }
        }
        Text("MATLAB과 nRF Connect의 기존 연결을 종료한 뒤 검색하세요.",color=Subtle,fontSize=12.sp)
        Row { TextButton(onClick=permissions) { Text("앱 권한 설정") };TextButton(onClick=bluetooth) { Text("Bluetooth 설정") } }
    }
}
@Composable private fun SensorGraph(title:String,samples:List<Sample>,channels:Int,fsr:Boolean,offset:Int=0) {
    val colors=listOf(Blue,Yellow,Color(0xFF7AD9C0),Color(0xFFF193B6),Color(0xFFBA9EFF),Color(0xFFFFAC74))
    Text(title,fontSize=13.sp)
    if(samples.isEmpty()) { Box(Modifier.fillMaxWidth().height(100.dp),contentAlignment=Alignment.Center) { Text("센서 데이터 대기",color=Subtle,fontSize=12.sp) };return }
    fun value(s:Sample,c:Int)=if(fsr)s.fsr[c] else s.imu[c+offset]
    val finite=samples.flatMap { s -> (0 until channels).map { value(s,it) }.filter(Double::isFinite) }
    val low=if(fsr)0.0 else min(-.1,finite.minOrNull()?:0.0)
    val high=if(fsr)4095.0 else max(.1,finite.maxOrNull()?:1.0)
    Canvas(Modifier.fillMaxWidth().height(110.dp).background(Background,RoundedCornerShape(10.dp))) {
        for(i in 1..3)drawLine(Subtle.copy(alpha=.12f),Offset(0f,size.height*i/4),Offset(size.width,size.height*i/4))
        val latest=samples.last().receivedElapsedMs
        for(c in 0 until channels) {
            val path=Path();var begun=false
            samples.forEach { s -> val v=value(s,c);if(v.isFinite()) {
                val x=(1-(latest-s.receivedElapsedMs)/10000.0).toFloat().coerceIn(0f,1f)*size.width
                val y=(1-(v-low)/(high-low)).toFloat().coerceIn(0f,1f)*size.height
                if(begun)path.lineTo(x,y) else {path.moveTo(x,y);begun=true}
            } else begun=false }
            drawPath(path,colors[c],style=androidx.compose.ui.graphics.drawscope.Stroke(width=2f))
        }
    }
    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) {
        Text("−10",fontSize=10.sp,color=Subtle);Text("경과 시간 (s)",fontSize=10.sp,color=Subtle);Text("0",fontSize=10.sp,color=Subtle)
    }
    if(fsr)Text("0–4,095 · 현재 최대값 ${(samples.last().fsr.maxOrNull()?:0.0).format(0)}",fontSize=10.sp,color=Subtle)
}
@Composable private fun NavIcon(index:Int,selected:Boolean) {
    val color=if(selected)Blue else Subtle
    Canvas(Modifier.size(27.dp).padding(3.dp)) {
        val w=size.width;val h=size.height
        when(index) {
            0,1 -> {
                val p=Path().apply { moveTo(w*.2f,h*.6f);lineTo(w*.2f,h*.23f);lineTo(w*.32f,h*.23f);lineTo(w*.32f,h*.06f);lineTo(w*.46f,h*.06f);lineTo(w*.46f,h*.33f);lineTo(w*.58f,h*.33f);lineTo(w*.58f,h*.16f);lineTo(w*.72f,h*.16f);lineTo(w*.72f,h*.55f);lineTo(w*.88f,h*.45f);lineTo(w*.95f,h*.58f);lineTo(w*.67f,h*.94f);lineTo(w*.35f,h*.94f);close() }
                drawPath(p,color,style=androidx.compose.ui.graphics.drawscope.Stroke(2f))
                if(index==1)drawCircle(color,2.5f,Offset(w*.46f,h*.64f))
            }
            2 -> { val p=Path().apply { moveTo(0f,h*.5f);lineTo(w*.25f,h*.5f);lineTo(w*.4f,h*.1f);lineTo(w*.6f,h*.9f);lineTo(w*.75f,h*.5f);lineTo(w,h*.5f) };drawPath(p,color,style=androidx.compose.ui.graphics.drawscope.Stroke(2.5f)) }
            3 -> { drawCircle(color,w*.4f,style=androidx.compose.ui.graphics.drawscope.Stroke(2f));drawCircle(color,w*.15f) }
            else -> { drawCircle(color,w*.3f,style=androidx.compose.ui.graphics.drawscope.Stroke(3f));for(i in 0..7) { val a=i*PI/4;drawLine(color,Offset(w/2+cos(a).toFloat()*w*.3f,h/2+sin(a).toFloat()*w*.3f),Offset(w/2+cos(a).toFloat()*w*.46f,h/2+sin(a).toFloat()*w*.46f),3f) } }
        }
    }
}
private fun Double.format(decimals:Int)=String.format(Locale.KOREA,"%.${decimals}f",this)
