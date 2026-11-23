package kr.ac.signai.offline.core

/** RFC4180 quoting including labels containing commas; labels never enter model features. */
object CsvReplay {
    fun records(text: String): List<List<String>> {
        val rows = mutableListOf<List<String>>(); val row = mutableListOf<String>(); val value=StringBuilder()
        var quoted=false; var i=0
        val input=text.removePrefix("\uFEFF")
        while(i<input.length) {
            val c=input[i]
            when {
                c=='"' && quoted && i+1<input.length && input[i+1]=='"' -> { value.append('"'); i++ }
                c=='"' -> quoted=!quoted
                c==',' && !quoted -> { row.add(value.toString()); value.clear() }
                (c=='\n'||c=='\r') && !quoted -> {
                    if(c=='\r' && i+1<input.length && input[i+1]=='\n')i++
                    row.add(value.toString());value.clear()
                    if(row.any { it.isNotEmpty() }) rows.add(row.toList());row.clear()
                }
                else -> value.append(c)
            };i++
        }
        require(!quoted) { "CSV 따옴표가 닫히지 않았습니다" }
        if(value.isNotEmpty()||row.isNotEmpty()) { row.add(value.toString());rows.add(row.toList()) }
        return rows
    }
    fun parse(text: String): Capture {
        val records=records(text);require(records.size>1) { "빈 CSV입니다" }
        val header=records.first();require(header.distinct().size==header.size)
        val required=listOf("time","source","hand","trial_id","label","time_ms","subject_id","session_id","sample_seq","imu_ok","board_type","record_kind")+
            (1..6).map { "fsr$it" }+listOf("ax","ay","az","gx","gy","gz")
        require(header.containsAll(required)) { "필수 CSV 열이 없습니다" }
        val identity=mutableMapOf<String,String>();val hands=Hand.entries.associateWith { linkedMapOf<Long,Sample>() }
        for(row in records.drop(1)) {
            require(row.size==header.size) { "CSV 열 수가 다릅니다" }
            fun field(key: String)=row[header.indexOf(key)].trim()
            require(field("source")=="REAL" && field("board_type")=="XIAO" && field("record_kind")=="training") { "실제 XIAO 학습 CSV만 재생할 수 있습니다" }
            for(key in listOf("subject_id","session_id","trial_id","label")) {
                val v=field(key);require(v.isNotEmpty() && (identity[key]==null || identity[key]==v)) { "서로 다른 회차가 섞여 있습니다" };identity[key]=v
            }
            val hand=when(field("hand")) { "left"->Hand.LEFT;"right"->Hand.RIGHT;else->error("손 정보 오류") }
            val seq=field("sample_seq").toLong()
            val sample=Sample(field("time_ms").toLong(),seq,DoubleArray(6) { field("fsr${it+1}").toDouble() },
                listOf("ax","ay","az","gx","gy","gz").map { field(it).toDouble() }.toDoubleArray(),field("imu_ok").toDouble()==1.0,
                java.time.Instant.parse(field("time")).toEpochMilli())
            val prior=hands.getValue(hand)[seq]
            require(prior==null || (prior.timeMs==sample.timeMs && prior.fsr.contentEquals(sample.fsr) && prior.imu.contentEquals(sample.imu) && prior.imuOk==sample.imuOk)) { "중복 번호의 센서값이 충돌합니다" }
            hands.getValue(hand).putIfAbsent(seq,sample)
        }
        return Capture(hands.getValue(Hand.LEFT).values.sortedBy { it.timeMs },hands.getValue(Hand.RIGHT).values.sortedBy { it.timeMs },true)
    }
}
