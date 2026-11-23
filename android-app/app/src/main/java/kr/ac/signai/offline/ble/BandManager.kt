@file:Suppress("DEPRECATION", "MissingPermission")
package kr.ac.signai.offline.ble

import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import kr.ac.signai.offline.core.*
import java.util.UUID

data class BandDevice(val address: String,val name: String,val rssi: Int,val hand: Hand?=null,
    val connected: Boolean=false,val ready: Boolean=false,val status: String="발견됨",val samples: Long=0,val missing: Long=0)

/** Two independent GATT clients; callbacks and all state changes are serialized on main. */
class BandManager(context: Context, private val changed: ()->Unit, private val received: (Hand,Sample)->Unit) {
    private val context=context.applicationContext
    private val preferences=context.getSharedPreferences("bands",Context.MODE_PRIVATE)
    private val handler=Handler(Looper.getMainLooper())
    private val adapter=(context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager).adapter
    val devices=linkedMapOf<String,BandDevice>()
    var scanning=false;private set
    var message="";private set
    var foreground=true;private set
    private data class Link(val address:String,val hand:Hand,val decoder:PacketDecoder=PacketDecoder(),var gatt:BluetoothGatt?=null,
        var enabled:Boolean=true,var attempts:Int=0,var lastElapsed:Long=0)
    private val links=mutableMapOf<String,Link>()
    private val service=UUID.fromString("19B10010-E8F2-537E-4F6C-D104768A1214")
    private val characteristic=UUID.fromString("19B10011-E8F2-537E-4F6C-D104768A1214")
    private val cccd=UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    fun available():Boolean = try { adapter?.isEnabled==true } catch(_:SecurityException) { false }
    fun bothReady():Boolean = Hand.entries.all { h -> devices.values.any { it.hand==h && it.ready } }
    fun fresh():Boolean = bothReady() && links.values.filter { it.enabled }.all { SystemClock.elapsedRealtime()-it.lastElapsed<1200 }
    private fun update(address:String,fn:(BandDevice)->BandDevice) { devices[address]?.let { devices[address]=fn(it);changed() } }
    fun assign(address:String, hand:Hand) {
        if(links[address]?.enabled==true) { message="손 지정을 바꾸려면 먼저 연결을 해제하세요";changed();return }
        preferences.edit().putString(address,hand.name).apply()
        update(address) { it.copy(hand=hand) }
    }
    private val scannerCallback=object:ScanCallback() {
        override fun onScanResult(type:Int,result:ScanResult) { handler.post {
            try {
                val device=result.device;val address=device.address
                val name=result.scanRecord?.deviceName?:device.name?:"이름 없는 기기"
                val prior=devices[address]
                val saved=preferences.getString(address,null)?.let { runCatching { Hand.valueOf(it) }.getOrNull() }
                devices[address]=(prior?:BandDevice(address,name,result.rssi,hand=saved)).copy(name=name,rssi=result.rssi)
                changed()
            } catch(_:SecurityException) { message="블루투스 권한이 필요합니다";stopScan() }
        } }
        override fun onScanFailed(errorCode:Int) { handler.post { scanning=false;message="검색 실패: $errorCode · 잠시 후 다시 검색하세요";changed() } }
    }
    fun scan() {
        if(scanning)return
        if(!available()) { message="Bluetooth가 꺼져 있거나 권한이 없습니다";changed();return }
        try {
            message="";scanning=true
            // Filter by advertised firmware service; names are display-only, never identify hands.
            val scanner=adapter.bluetoothLeScanner ?: error("BLE 검색을 사용할 수 없습니다")
            scanner.startScan(listOf(ScanFilter.Builder().setServiceUuid(android.os.ParcelUuid(service)).build()),
                ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build(),scannerCallback)
            changed();handler.postDelayed({stopScan()},12000)
        } catch(e:Exception) { scanning=false;message="검색 실패: ${e.message}";changed() }
    }
    fun stopScan() { if(scanning)runCatching { adapter?.bluetoothLeScanner?.stopScan(scannerCallback) };scanning=false;changed() }
    fun connect(address:String) {
        val hand=devices[address]?.hand
        if(hand==null) { message="먼저 왼손 또는 오른손을 지정하세요";changed();return }
        if(links.values.any { it.enabled && it.hand==hand && it.address!=address }) {
            message="${hand.korean}에 다른 밴드가 연결되어 있습니다. 먼저 연결을 해제하세요";changed();return
        }
        if(links[address]?.enabled==true)return
        val link=Link(address,hand);links[address]=link;open(link)
    }
    fun disconnect(address:String) {
        links.remove(address)?.let { it.enabled=false;runCatching { it.gatt?.disconnect();it.gatt?.close() };it.gatt=null }
        update(address) { it.copy(connected=false,ready=false,status="연결 해제됨") }
    }
    private fun open(link:Link) {
        if(!link.enabled || !foreground)return
        if(!available()) { fail(link,"Bluetooth가 꺼져 있습니다");return }
        update(link.address) { it.copy(connected=false,ready=false,status=if(link.attempts==0)"연결 중" else "자동 재연결 중") }
        link.decoder.reset();link.lastElapsed=0
        try { link.gatt=adapter.getRemoteDevice(link.address).connectGatt(context,false,callback(link),BluetoothDevice.TRANSPORT_LE) }
        catch(e:Exception) { fail(link,e.message?:"GATT 연결 실패") }
        val connection=link.gatt ?: return
        handler.postDelayed({ if(link.enabled && link.gatt===connection && devices[link.address]?.ready!=true)fail(link,"연결 또는 알림 설정 시간 초과") },15000)
    }
    private fun current(link:Link,gatt:BluetoothGatt)=link.enabled && links[link.address]===link && link.gatt===gatt
    private fun fail(link:Link,why:String) {
        if(!link.enabled)return
        runCatching { link.gatt?.disconnect();link.gatt?.close() };link.gatt=null
        update(link.address) { it.copy(connected=false,ready=false,status="연결 끊김: $why") }
        link.attempts++
        val wait=(1000L shl link.attempts.coerceAtMost(5)).coerceAtMost(30000)
        handler.postDelayed({if(link.enabled && foreground && link.gatt==null)open(link)},wait)
    }
    private fun callback(link:Link)=object:BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt:BluetoothGatt,status:Int,newState:Int) { handler.post {
            if(!current(link,gatt))return@post
            if(status!=BluetoothGatt.GATT_SUCCESS || newState==BluetoothProfile.STATE_DISCONNECTED) { fail(link,"GATT $status");return@post }
            if(newState==BluetoothProfile.STATE_CONNECTED) {
                update(link.address) { it.copy(connected=true,status="서비스 확인 중") }
                try { gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH);if(!gatt.discoverServices())fail(link,"서비스 검색 시작 실패") }
                catch(e:SecurityException) { fail(link,"권한이 취소되었습니다") }
            }
        } }
        override fun onServicesDiscovered(gatt:BluetoothGatt,status:Int) { handler.post {
            if(!current(link,gatt))return@post
            try {
                val ch=gatt.getService(service)?.getCharacteristic(characteristic)
                val descriptor=ch?.getDescriptor(cccd)
                if(status!=BluetoothGatt.GATT_SUCCESS || ch==null || descriptor==null || !gatt.setCharacteristicNotification(ch,true)) {
                    fail(link,"펌웨어 서비스 또는 Notify 설정 실패");return@post
                }
                val started=if(Build.VERSION.SDK_INT>=33) gatt.writeDescriptor(descriptor,BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)==BluetoothStatusCodes.SUCCESS
                    else { descriptor.value=BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE;gatt.writeDescriptor(descriptor) }
                if(!started)fail(link,"CCCD 쓰기 시작 실패")
            } catch(e:Exception) { fail(link,e.message?:"알림 설정 실패") }
        } }
        override fun onDescriptorWrite(gatt:BluetoothGatt,descriptor:BluetoothGattDescriptor,status:Int) { handler.post {
            if(!current(link,gatt))return@post
            if(status==BluetoothGatt.GATT_SUCCESS && descriptor.uuid==cccd) {
                link.attempts=0;update(link.address) { it.copy(ready=true,status="센서 데이터 대기") }
            } else fail(link,"CCCD 응답 $status")
        } }
        private fun notify(gatt:BluetoothGatt,value:ByteArray) {
            val copy=value.copyOf();val utc=System.currentTimeMillis();val elapsed=SystemClock.elapsedRealtime()
            handler.post {
                if(!current(link,gatt))return@post
                val sample=link.decoder.accept(copy,utc,elapsed)
                if(sample!=null) { link.lastElapsed=elapsed;received(link.hand,sample) }
                update(link.address) { it.copy(status=if(sample!=null)"데이터 수신 중" else it.status,samples=link.decoder.samples,missing=link.decoder.missing) }
            }
        }
        override fun onCharacteristicChanged(gatt:BluetoothGatt,ch:BluetoothGattCharacteristic,value:ByteArray) { if(ch.uuid==characteristic)notify(gatt,value) }
        override fun onCharacteristicChanged(gatt:BluetoothGatt,ch:BluetoothGattCharacteristic) { if(Build.VERSION.SDK_INT<33 && ch.uuid==characteristic)notify(gatt,ch.value?:byteArrayOf()) }
    }
    fun setForeground(value:Boolean) { foreground=value;if(!value)stopScan() else links.values.filter { it.enabled && it.gatt==null }.forEach { open(it) } }
    fun close() { stopScan();links.keys.toList().forEach { disconnect(it) };handler.removeCallbacksAndMessages(null) }
}
