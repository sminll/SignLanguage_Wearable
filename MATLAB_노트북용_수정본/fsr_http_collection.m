function collector = fsr_http_collection(fig, saveDirectory)
% Keep a CSV backup while collecting. POST only after the user presses Send.
if nargin < 2
    saveDirectory = fullfile(fileparts(mfilename('fullpath')), 'sensor_data');
end
endpoint = 'http://100.92.15.45:5000/sensor'; % Remote receiver over Tailscale
running = false;
sending = false;
submitted = false;
hasRun = false;
complete = false;
clockInvalid = false;
startMs = NaN;
lastMs = NaN;
elapsedMs = 0;
elapsedOffset = 0;
gestureName = '';
subjectId = '';
sessionId = '';
deviceId = '';
sequence = 0;
sent = 0;
unconfirmed = 0;
trialCounts = zeros(1, 20);
phase = sensor_collection_phase(0);
pending = {};
sessionSamples = {};
csvBuffer = {}; % Batch disk writes instead of opening CSV for every frame.
csvError = false;
lastCsvFlush = tic;
csvPath = '';
activeBatch = {};
transfer = [];
message = '수집 데이터는 CSV에도 자동 저장합니다. 확인 후 전송 버튼을 누르세요.';

uicontrol(fig, 'Style', 'text', 'String', '동작명', ...
    'Units', 'normalized', 'Position', [0.02 0.95 0.07 0.03], ...
    'BackgroundColor', 'w', 'ForegroundColor', [0.15 0.15 0.15]);
gestureInput = uicontrol(fig, 'Style', 'edit', 'String', '', ...
    'Units', 'normalized', 'Position', [0.09 0.944 0.15 0.04], ...
    'Tag', 'SensorGestureName', 'BackgroundColor', 'w', ...
    'ForegroundColor', [0.15 0.15 0.15]);
handInput = uicontrol(fig, 'Style', 'popupmenu', ...
    'String', {'손 선택', '왼손 (L)', '오른손 (R)'}, 'Value', 1, ...
    'Units', 'normalized', 'Position', [0.25 0.944 0.13 0.04], ...
    'Tag', 'SensorHand', 'BackgroundColor', 'w', ...
    'ForegroundColor', [0.15 0.15 0.15]);
startButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '20회 수집 시작', ...
    'Units', 'normalized', 'Position', [0.39 0.944 0.15 0.04], ...
    'Tag', 'SensorStart', 'Callback', @startCollection);
stopButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '중지', ...
    'Units', 'normalized', 'Position', [0.55 0.944 0.08 0.04], ...
    'Tag', 'SensorStop', 'Callback', @stopCollection);
% Participant names appear only in the UI. CSV stores e.g. 'subject3'.
subjectOptions = {'측정자 선택', 'subject1 (정성윤)', 'subject2 (임영훈)', ...
    'subject3 (주다경)', 'subject4 (고준혁)', 'subject5 (지상민)', ...
    'subject6 (이수민)', 'subject7 (박경민)', 'subject8 (백수현)', ...
    'subject9 (이하영)', 'subject10 (정소윤)'};
subjectInput = uicontrol(fig, 'Style', 'popupmenu', 'String', subjectOptions, ...
    'Value', 1, 'Units', 'normalized', 'Position', [0.02 0.892 0.25 0.04], ...
    'Tag', 'SensorSubject');
deleteButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '최근 회차 삭제', ...
    'Units', 'normalized', 'Position', [0.28 0.892 0.12 0.04], ...
    'Tag', 'SensorDeleteLast', 'Callback', @deleteLastTrial);
clearButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '전체 삭제', ...
    'Units', 'normalized', 'Position', [0.41 0.892 0.11 0.04], ...
    'Tag', 'SensorClear', 'Callback', @clearCollection);
sendButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '전송', ...
    'Units', 'normalized', 'Position', [0.53 0.892 0.12 0.04], ...
    'Tag', 'SensorSend', 'Callback', @sendCollection);
correctButton = uicontrol(fig, 'Style', 'pushbutton', 'String', '정보 수정', ...
    'Units', 'normalized', 'Position', [0.66 0.892 0.12 0.04], ...
    'Tag', 'SensorCorrect', 'Callback', @correctCollection);
csvLabel = uicontrol(fig, 'Style', 'text', 'String', 'CSV: 수집 시작 시 자동 생성', ...
    'Units', 'normalized', 'Position', [0.79 0.892 0.20 0.035], ...
    'Tag', 'SensorCSVFile', 'BackgroundColor', 'w', ...
    'ForegroundColor', [0.15 0.15 0.15], 'HorizontalAlignment', 'left');
statusLabel = uicontrol(fig, 'Style', 'text', 'String', message, ...
    'Units', 'normalized', 'Position', [0.02 0.845 0.96 0.035], ...
    'Tag', 'SensorStatus', 'BackgroundColor', 'w', ...
    'ForegroundColor', [0.15 0.15 0.15], 'HorizontalAlignment', 'left');

collector = struct('addSample', @addSample, 'tick', @tick, 'close', @closeCollector);
tick();

    function startCollection(~, ~)
        if running || sending || ~isempty(transfer) || ...
                (hasRun && (complete || submitted || clockInvalid))
            return;
        end
        if ~hasRun
            gestureName = strtrim(get(gestureInput, 'String'));
            handIndex = get(handInput, 'Value');
            subjectIndex = get(subjectInput, 'Value');
            if isempty(gestureName) || handIndex == 1 || subjectIndex == 1
                message = 'Subject, 동작명, 손을 모두 선택하세요.';
                tick();
                return;
            end
            if ~isempty(regexp(gestureName, '[0-9]$', 'once')) || ...
                    ~isempty(regexp(gestureName, '[<>:"/\\|?*]', 'once'))
                message = '수어명은 숫자로 끝나거나 파일명 특수문자를 포함할 수 없습니다.';
                tick();
                return;
            end
            subjectId = sprintf('subject%d', subjectIndex - 1);
            subjectLabel = subjectOptions{subjectIndex};
            hands = {'L', 'R'};
            deviceId = hands{handIndex - 1};
            if strcmp(get(fig, 'Visible'), 'on')
                summary = sprintf('%s / %s / %s / 20회 측정', subjectLabel, gestureName, deviceId);
                answer = questdlg([summary newline '측정자가 정확한지 확인해 주세요.'], ...
                    '측정 전 확인', '맞습니다', '취소', '취소');
                if ~strcmp(answer, '맞습니다'), return; end
            end
            try
                if ~isfolder(saveDirectory)
                    [ok, detail] = mkdir(saveDirectory);
                    if ~ok
                        error('SensorCSV:Directory', '%s', detail);
                    end
                end
                safeName = regexprep(gestureName, '[<>:"/\\|?*\x00-\x1F]', '_');
                stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
                sessionId = sprintf('s%s_%09d', stamp, randi(999999999));
                baseName = sprintf('%s_%s_%s_%s', subjectId, safeName, deviceId, stamp);
                newPath = fullfile(saveDirectory, [baseName '.csv']);
                suffix = 1;
                while isfile(newPath) || isfolder(newPath)
                    newPath = fullfile(saveDirectory, sprintf('%s_%d.csv', baseName, suffix));
                    suffix = suffix + 1;
                end
                write_sensor_csv(newPath, {}, 'replace');
            catch err
                message = ['CSV 파일을 만들지 못해 수집을 시작하지 않았습니다: ' err.message];
                tick();
                return;
            end
            csvPath = newPath;
            sessionSamples = {};
            csvBuffer = {};
            csvError = false;
            lastCsvFlush = tic;
            [~, fileName, extension] = fileparts(csvPath);
            set(csvLabel, 'String', ['CSV: ' fileName extension], 'TooltipString', csvPath);
            hasRun = true;
            submitted = false;
            complete = false;
            clockInvalid = false;
            elapsedMs = 0;
            sent = 0;
            unconfirmed = 0;
            trialCounts(:) = 0;
            phase = sensor_collection_phase(0);
        else
            try
                write_sensor_csv(csvPath, sessionSamples, 'replace');
            catch err
                message = ['CSV 저장 오류로 재개하지 않았습니다: ' err.message];
                tick();
                return;
            end
        end
        elapsedOffset = elapsedMs;
        startMs = NaN; % Resume from the next frame, excluding paused time.
        running = true;
        message = '';
        tick();
    end

    function stopCollection(~, ~)
        if ~running
            return;
        end
        running = false;
        flushCsvBuffer();
        if ~csvError
            message = sprintf('%d/20회차에서 중지. 보관 데이터를 삭제하거나 계속 수집할 수 있습니다.', phase.trial);
        end
        tick();
    end

    function deleteLastTrial(~, ~)
        if running || sending || submitted
            return;
        end
        trial = find(trialCounts > 0, 1, 'last');
        if isempty(trial)
            return;
        end
        source = sprintf('%s%d', gestureName, trial);
        remove = cellfun(@(sample) strcmp(sample.source, source), pending);
        keep = ~cellfun(@(sample) strcmp(sample.source, source), sessionSamples);
        try
            write_sensor_csv(csvPath, sessionSamples(keep), 'replace');
            csvBuffer = {};
            csvError = false;
        catch err
            message = ['CSV 갱신 실패로 데이터를 삭제하지 않았습니다: ' err.message];
            tick();
            return;
        end
        pending(remove) = [];
        sessionSamples = sessionSamples(keep);
        trialCounts(trial) = 0;
        elapsedMs = (trial - 1) * 8000;
        elapsedOffset = elapsedMs;
        startMs = NaN;
        phase = sensor_collection_phase(elapsedMs);
        complete = false;
        if clockInvalid
            message = sprintf('%s 삭제. 보드가 재시작되어 새 수집은 전체 삭제 후 시작하세요.', source);
        else
            message = sprintf('%s 삭제. 계속 수집을 누르면 %d회차부터 다시 수집합니다.', source, trial);
        end
        tick();
    end

    function clearCollection(~, ~)
        if running || sending || ~isempty(transfer)
            return;
        end
        count = numel(pending);
        if hasRun && ~submitted
            try
                write_sensor_csv(csvPath, {}, 'replace');
            catch err
                message = ['CSV 갱신 실패로 데이터를 삭제하지 않았습니다: ' err.message];
                tick();
                return;
            end
        end
        if submitted
            message = '메모리 보관을 정리했습니다. 전송 시도한 데이터의 CSV는 유지합니다.';
        else
            message = sprintf('보관 데이터 %d개 삭제(CSV 반영). 새로 수집할 수 있습니다.', count);
        end
        pending = {};
        sessionSamples = {};
        csvBuffer = {};
        csvError = false;
        hasRun = false;
        complete = false;
        submitted = false;
        clockInvalid = false;
        elapsedMs = 0;
        elapsedOffset = 0;
        trialCounts(:) = 0;
        tick();
    end

    function correctCollection(~, ~)
        if running || sending || submitted || ~hasRun || isempty(sessionSamples)
            return;
        end
        nameNew = strtrim(get(gestureInput, 'String'));
        indexNew = get(subjectInput, 'Value');
        if isempty(nameNew) || indexNew == 1 || ...
                ~isempty(regexp(nameNew, '[0-9]$', 'once')) || ...
                ~isempty(regexp(nameNew, '[<>:"/\\|?*]', 'once'))
            message = '새 수어명과 측정자 번호를 확인해 주세요.';
            tick(); return;
        end
        subjectNew = sprintf('subject%d', indexNew - 1);
        if strcmp(nameNew, gestureName) && strcmp(subjectNew, subjectId)
            message = '현재 Subject와 수어 이름이 이미 일치합니다.';
            tick(); return;
        end
        if strcmp(get(fig, 'Visible'), 'on')
            prompt = sprintf('%s / %s -> %s / %s\n전체 %d행을 수정합니다. 맞나요?', ...
                subjectId, gestureName, subjectNew, nameNew, numel(sessionSamples));
            answer = questdlg(prompt, '측정 정보 수정', '수정', '취소', '취소');
            if ~strcmp(answer, '수정'), return; end
        end
        revised = sessionSamples;
        for j = 1:numel(revised)
            copy = revised{j};
            suffix = regexp(copy.source, '[0-9]+$', 'match', 'once');
            if isempty(suffix), message = '회차명 오류로 수정하지 않았습니다.'; tick(); return; end
            copy.source = [nameNew suffix];
            copy.subject_id = subjectNew;
            revised{j} = copy;
        end
        try
            write_sensor_csv(csvPath, revised, 'replace');
        catch err
            message = ['CSV 수정 실패: ' err.message];
            tick(); return;
        end
        sessionSamples = revised;
        pending = revised;
        csvBuffer = {};
        gestureName = nameNew;
        subjectId = subjectNew;
        message = sprintf('%s / %s 로 수정 완료. 파일명은 기존 이름이며 CSV 내부 정보가 정정됐습니다.', ...
            subjectId, gestureName);
        tick();
    end

    function sendCollection(~, ~)
        if running || sending || submitted || isempty(pending)
            return;
        end
        if ~strcmp(subjectId, sprintf('subject%d', get(subjectInput, 'Value')-1)) || ...
                ~strcmp(gestureName, strtrim(get(gestureInput, 'String')))
            message = '측정 정보가 바뀌었습니다. 먼저 [정보 수정]을 눌러 저장하세요.';
            tick(); return;
        end
        if strcmp(get(fig, 'Visible'), 'on')
            prompt = sprintf('Subject: %s\n수어: %s\n손: %s\n측정 행 수: %d\n전송 후에는 MATLAB에서 정보 수정 불가', ...
                subjectOptions{get(subjectInput, 'Value')}, gestureName, deviceId, numel(pending));
            answer = questdlg(prompt, '전송 전 최종 확인', '전송', '취소', '취소');
            if ~strcmp(answer, '전송'), return; end
        end
        try
            write_sensor_csv(csvPath, sessionSamples, 'replace');
            csvBuffer = {};
            csvError = false;
        catch err
            message = ['CSV 저장 실패로 전송을 시작하지 않았습니다: ' err.message];
            tick();
            return;
        end
        submitted = true;
        sending = true;
        message = '확인한 보관 데이터를 전송하는 중입니다.';
        tick();
    end

    function addSample(milliseconds, fsr, imu)
        if hasRun && ~submitted && ~isnan(lastMs) && milliseconds < lastMs
            running = false;
            clockInvalid = true;
            message = 'ESP32 재시작: 계속 수집할 수 없습니다. 보관 데이터를 확인하세요.';
        end
        lastMs = milliseconds;
        if ~running
            return;
        end
        if isnan(startMs)
            startMs = milliseconds;
        end
        elapsedMs = elapsedOffset + milliseconds - startMs;
        phase = sensor_collection_phase(elapsedMs);
        if phase.done
            running = false;
            complete = all(trialCounts > 0);
            if ~complete
                message = '데이터가 없는 회차가 있습니다. 보관 데이터를 확인하세요.';
            end
            return;
        end
        if phase.collecting
            sequence = sequence + 1;
            source = sprintf('%s%d', gestureName, phase.trial);
            pending{end + 1} = make_sensor_payload( ...
                fsr, imu, deviceId, sequence, milliseconds, source, subjectId, sessionId);
            sessionSamples{end + 1} = pending{end};
            trialCounts(phase.trial) = trialCounts(phase.trial) + 1;
            csvBuffer{end + 1} = pending{end};
            if numel(csvBuffer) >= 20
                flushCsvBuffer();
            end
        end
    end

    function tick()
        if ~isgraphics(statusLabel)
            return;
        end
        % GUI timer call: flush buffered frames to CSV in a single write.
        if ~csvError && ~isempty(csvBuffer) && (toc(lastCsvFlush) >= 1.0 || ...
                strcmp(get(fig, 'Visible'), 'off'))
            flushCsvBuffer();
        end
        if ~isempty(transfer) && strcmp(transfer.State, 'finished')
            try
                result = fetchOutputs(transfer);
            catch err
                result = struct('sent', 0, 'unconfirmed', numel(activeBatch), ...
                    'message', err.message);
            end
            delete(transfer);
            transfer = [];
            sent = sent + result.sent;
            if result.unconfirmed > 0
                pending = [activeBatch(result.sent + 1:end) pending];
                unconfirmed = unconfirmed + result.unconfirmed;
                sending = false;
                message = ['HTTP 오류: 잔여 데이터 보존. 자동 재전송하지 않습니다. ' result.message];
            end
            activeBatch = {};
        end
        if sending && isempty(transfer) && ~isempty(pending)
            count = min(20, numel(pending));
            activeBatch = pending(1:count);
            pending(1:count) = [];
            try
                transfer = parfeval(backgroundPool, @post_sensor_batch, 1, endpoint, activeBatch);
            catch err
                pending = [activeBatch pending];
                unconfirmed = unconfirmed + numel(activeBatch);
                activeBatch = {};
                sending = false;
                message = ['HTTP 작업 시작 실패: 보관 데이터 유지. ' err.message];
            end
        end
        if sending && isempty(transfer) && isempty(pending)
            sending = false;
            hasRun = false;
            complete = false;
            message = sprintf('보관 데이터 전송 완료: %d개 확인.', sent);
        end
        if running && isnan(startMs)
            state = '첫 센서 데이터를 기다리는 중';
        elseif running
            if phase.collecting
                action = '수집';
            else
                action = '이완';
            end
            state = sprintf('%s%d | %s %d/20 | 남은 %.1f초', ...
                gestureName, phase.trial, action, phase.trial, phase.remainingSec);
        elseif sending
            state = '보관 데이터 전송 중';
        elseif submitted || ~isempty(message)
            state = message;
        elseif complete
            state = '20회 수집 완료. 확인 후 전송 버튼을 누르세요.';
        else
            state = message;
        end
        set(statusLabel, 'String', sprintf('%s | 보관 %d | 전송 중 %d | 전송 확인 %d | 미확인 %d', ...
            state, numel(pending), numel(activeBatch), sent, unconfirmed));
        idle = ~running && ~sending;
        editable = idle && ~hasRun && isempty(pending);
        canResume = hasRun && ~phase.done && ~complete && ~submitted && ~clockInvalid;
        % Subject and gesture can be edited while stopped, then committed
        % with [정보 수정]. Hand cannot change after a run starts.
        setEnabled([gestureInput subjectInput], idle && (~hasRun || ~submitted));
        setEnabled(handInput, editable);
        setEnabled(correctButton, idle && hasRun && ~submitted && ~isempty(sessionSamples));
        setEnabled(startButton, idle && (editable || canResume));
        if hasRun
            set(startButton, 'String', '계속 수집');
        else
            set(startButton, 'String', '20회 수집 시작');
        end
        setEnabled(stopButton, running);
        setEnabled(deleteButton, idle && ~submitted && any(trialCounts > 0));
        setEnabled(clearButton, idle && (hasRun || ~isempty(pending)));
        if submitted && hasRun
            set(clearButton, 'String', '새 수집');
        else
            set(clearButton, 'String', '전체 삭제');
        end
        setEnabled(sendButton, idle && ~submitted && ~isempty(pending));
    end

    function flushCsvBuffer()
        if isempty(csvBuffer) || csvError
            return;
        end
        try
            write_sensor_csv(csvPath, csvBuffer, 'append');
            csvBuffer = {};
            lastCsvFlush = tic;
        catch err
            csvError = true;
            running = false;
            message = ['CSV 저장 오류로 수집 중지(메모리 데이터 유지): ' err.message];
        end
    end

    function closeCollector()
        if ~isempty(transfer)
            cancel(transfer);
            delete(transfer);
            transfer = [];
        end
        if ~isempty(csvPath) && ~isempty(sessionSamples)
            try
                write_sensor_csv(csvPath, sessionSamples, 'replace');
                csvBuffer = {};
                fprintf('CSV 저장: %s\n', csvPath);
            catch err
                warning('SensorCSV:CloseWrite', '종료 시 CSV 저장 실패: %s', err.message);
            end
        end
        if ~isempty(activeBatch)
            warning('SensorHTTP:UnconfirmedOnClose', ...
                '전송 중 %d개는 서버 저장 여부를 확인하지 못했습니다. CSV 보관 파일을 확인하세요.', ...
                numel(activeBatch));
        end
    end
end

function setEnabled(controls, enabled)
states = {'off', 'on'};
value = states{double(enabled) + 1};
for k = 1:numel(controls)
    if ~strcmp(get(controls(k), 'Enable'), value)
        set(controls(k), 'Enable', value);
    end
end
end
