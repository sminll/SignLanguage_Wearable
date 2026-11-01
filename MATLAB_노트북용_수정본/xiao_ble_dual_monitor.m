function xiao_ble_dual_monitor(leftAddress, rightAddress)
% XIAO_BLE_DUAL_MONITOR - two XIAO nRF52840 Sense BLE devices, one session.
%
%   blelist
%   xiao_ble_dual_monitor("LEFT_BLE_ADDRESS", "RIGHT_BLE_ADDRESS")
%
% Keep each physical board's address assigned to its actual hand. Both may
% advertise as XIAO-FSR-IMU, so DO NOT connect twice by device name.
% 5 s capture + 3 s rest, 20 cycles, ONE shared host-side trial clock.
% CSV files are separate by hand and share subject_id / session_id.
% Requires: MATLAB BLE functions + existing make_sensor_payload.m,
%           write_sensor_csv.m, post_sensor_batch.m, sensor_collection_phase.m
% NOTE: hardware and parallel connection have NOT yet been validated.

if nargin ~= 2 || strlength(string(leftAddress)) == 0 || ...
        strlength(string(rightAddress)) == 0
    error('XIAO:Addresses', ['먼저 MATLAB에서 blelist를 실행해 두 장치 주소를 확인하세요. ' ...
        '그 후 xiao_ble_dual_monitor("왼쪽주소","오른쪽주소")를 입력하세요.']);
end
leftAddress = char(string(leftAddress));
rightAddress = char(string(rightAddress));
if strcmpi(leftAddress, rightAddress)
    error('XIAO:Addresses', '왼쪽/오른쪽에 서로 다른 BLE 주소를 지정해야 합니다.');
end
required = {'xiao_ble_packet_decode', 'make_sensor_payload', ...
    'write_sensor_csv', 'post_sensor_batch', 'sensor_collection_phase'};
for k = 1:numel(required)
    if exist(required{k}, 'file') ~= 2
        error('XIAO:MissingFile', '%s.m 파일을 MATLAB Current Folder에 넣어주세요.', required{k});
    end
end

serviceUUID = '19B10010-E8F2-537E-4F6C-D104768A1214';
characteristicUUID = '19B10011-E8F2-537E-4F6C-D104768A1214';
endpoint = 'http://100.92.15.45:5000/sensor';
saveDirectory = fullfile(fileparts(mfilename('fullpath')), 'sensor_data');
windowSeconds = 20;
addresses = {leftAddress, rightAddress};
hands = {'L', 'R'};
handNames = {'왼손', '오른손'};
subjectOptions = {'측정자 선택', 'subject1 (정성윤)', 'subject2 (임영훈)', ...
    'subject3 (주다경)', 'subject4 (고준혁)', 'subject5 (지상민)', ...
    'subject6 (이수민)', 'subject7 (박경민)', 'subject8 (백수현)', ...
    'subject9 (이하영)', 'subject10 (정소윤)'};

% Independent receiving/decoding state for both BLE connections.
devices = cell(1,2);
chars = cell(1,2);
fragments = cell(1,2);
lastFrame = cell(1,2);
lastSeen = cell(1,2);
lastDeviceMs = [NaN NaN];
firstPlotMs = [NaN NaN];
lastPlotMs = [NaN NaN];
received = zeros(1,2);
previousFrameSeq = [NaN NaN];
missingFrames = zeros(1,2);

% A single host clock keeps the left/right trial number consistent.
running = false;
completed = false;
submitted = false;
sending = false;
hasRun = false;
clock = [];
elapsedOffsetMs = 0;
elapsedMs = 0;
gestureName = '';
subjectId = '';
sessionId = '';
sequence = zeros(1,2); % Separate sample numbers within each hand.
trialCounts = zeros(2,20);
filePaths = {'', ''};
samples = {cell(1,0), cell(1,0)};
buffers = {cell(1,0), cell(1,0)};
pending = cell(1,0);
activeBatch = cell(1,0);
transfer = [];
sent = 0;
lastFlush = tic;
message = '먼저 두 XIAO에서 센서값이 수신되는지 확인하세요.';

fig = figure('Name', 'XIAO BLE 양손 센서 모니터 (동시 수신)', ...
    'NumberTitle', 'off', 'Color', 'w', 'Position', [70 70 1300 860], ...
    'Tag', 'XIAODualBLE', 'CloseRequestFcn', @(src,~) delete(src));
cleaner = onCleanup(@cleanupConnection); %#ok<NASGU>

uicontrol(fig, 'Style','text', 'String','수어명', 'Units','normalized', ...
    'Position',[0.02 0.955 0.06 0.03], 'BackgroundColor','w');
gestureInput = uicontrol(fig, 'Style','edit', 'String','', 'Units','normalized', ...
    'Position',[0.085 0.95 0.15 0.045], 'BackgroundColor','w');
subjectInput = uicontrol(fig, 'Style','popupmenu', 'String',subjectOptions, ...
    'Value',1, 'Units','normalized', 'Position',[0.25 0.95 0.22 0.045], ...
    'BackgroundColor','w');
startButton = uicontrol(fig,'Style','pushbutton','String','양손 20회 수집 시작', ...
    'Units','normalized','Position',[0.49 0.95 0.16 0.045], 'Callback',@startCapture);
stopButton = uicontrol(fig,'Style','pushbutton','String','일시 중지', ...
    'Units','normalized','Position',[0.67 0.95 0.10 0.045], 'Callback',@stopCapture);
correctButton = uicontrol(fig,'Style','pushbutton','String','정보 수정', ...
    'Units','normalized','Position',[0.79 0.95 0.10 0.045], 'Callback',@correctMetadata);
clearButton = uicontrol(fig,'Style','pushbutton','String','새 수집/초기화', ...
    'Units','normalized','Position',[0.90 0.95 0.09 0.045], 'Callback',@resetCapture);
deleteButton = uicontrol(fig,'Style','pushbutton','String','최근 회차 삭제', ...
    'Units','normalized','Position',[0.02 0.895 0.14 0.045], 'Callback',@deleteLastTrial);
sendButton = uicontrol(fig,'Style','pushbutton','String','Python 서버로 전송', ...
    'Units','normalized','Position',[0.18 0.895 0.17 0.045], 'Callback',@sendCapture);
% FSR display scaling (0..selected maximum). Does not modify measured ADC values.
fsrYMaxValues = [50 100 200 500 1000 4095];
uicontrol(fig,'Style','text','String','Y max (ADC)', ...
    'Units','normalized','Position',[0.75 0.895 0.11 0.04], ...
    'BackgroundColor','w','HorizontalAlignment','right');
fsrYMaxPopup = uicontrol(fig,'Style','popupmenu', ...
    'String',cellstr(string(fsrYMaxValues)), ...
    'Value',numel(fsrYMaxValues),'Units','normalized', ...
    'Position',[0.87 0.895 0.11 0.045], ...
    'Callback',@changeFsrYMax);
statusLabel = uicontrol(fig,'Style','text','String',message, ...
    'Units','normalized','Position',[0.02 0.845 0.96 0.035], ...
    'BackgroundColor','w','HorizontalAlignment','left', ...
    'ForegroundColor',[0.15 0.15 0.15]);

panel = uipanel(fig, 'BorderType','none','BackgroundColor','w', ...
    'Units','normalized','Position',[0 0.015 1 0.82]);
layout = tiledlayout(panel,2,3,'TileSpacing','compact','Padding','compact');
axesHandles = gobjects(2,3);
curves = cell(2,3);
colors = lines(6);
for side = 1:2
    for col = 1:3
        ax = nexttile(layout,(side-1)*3+col);
        axesHandles(side,col) = ax;
        hold(ax,'on'); grid(ax,'on');
        set(ax,'Color','w'); xlabel(ax,'Time (s)');
        xlim(ax,[0 windowSeconds]);
        if col == 1
            title(ax,sprintf('%s FSR (A0~A5)',handNames{side}));
            ylabel(ax,'ADC raw'); ylim(ax,[0 fsrYMaxValues(end)]);
            labels = arrayfun(@(k) sprintf('FSR%d',k),1:6,'UniformOutput',false);
        elseif col == 2
            title(ax,sprintf('%s IMU 가속도',handNames{side}));
            ylabel(ax,'g'); labels = {'AX','AY','AZ'};
        else
            title(ax,sprintf('%s IMU 자이로',handNames{side}));
            ylabel(ax,'deg/s'); labels = {'GX','GY','GZ'};
        end
        linesForAxis = gobjects(1,numel(labels));
        for c = 1:numel(labels)
            linesForAxis(c) = animatedline(ax,'Color',colors(c,:), ...
                'LineWidth',1.2,'MaximumNumPoints',1200);
        end
        curves{side,col} = linesForAxis;
        legend(ax,linesForAxis,labels,'Location','northeast', ...
            'AutoUpdate','off','FontSize',7);
    end
end

try
    fprintf('[XIAO BLE] 왼손 연결: %s\n', leftAddress);
    devices{1} = ble(leftAddress);
    fprintf('[XIAO BLE] 오른손 연결: %s\n', rightAddress);
    devices{2} = ble(rightAddress);
    for side = 1:2
        chars{side} = characteristic(devices{side},serviceUUID,characteristicUUID);
        chars{side}.DataAvailableFcn = @(src,~) receiveFragment(src,side);
        subscribe(chars{side},'notification');
        fprintf('[XIAO BLE] %s 알림 구독 완료.\n',handNames{side});
    end
    message = '양쪽 BLE 연결 완료. 손가락으로 각 FSR을 눌러 왼손/오른손 대응을 확인하세요.';
    refreshUI();
    updateClock = tic;
    while isgraphics(fig)
        if toc(updateClock) >= 0.25
            refreshUI();
            drawnow limitrate;
            updateClock = tic;
        end
        pause(0.02); % Processes BLE notification callbacks and UI events.
    end
catch exception
    if isgraphics(fig), delete(fig); end
    rethrow(exception);
end

    function receiveFragment(src,side)
        if ~isgraphics(fig), return; end
        try
            bytes = read(src,'oldest');
            [fragments{side},frame] = xiao_ble_packet_decode(fragments{side},bytes);
            if isempty(frame), return; end
            received(side) = received(side)+1;
            lastFrame{side} = frame;
            lastSeen{side} = tic;
            if ~isnan(previousFrameSeq(side))
                diffSeq = mod(frame.sequence-previousFrameSeq(side),2^32);
                if diffSeq > 1 && diffSeq < 2^31
                    missingFrames(side) = missingFrames(side)+diffSeq-1;
                end
            end
            previousFrameSeq(side) = frame.sequence;

            if ~isnan(lastDeviceMs(side)) && frame.time_ms < lastDeviceMs(side)
                % A reboot invalidates a continuous acquisition session.
                if running
                    running = false;
                    flushBuffers();
                    message = sprintf('%s XIAO 시간 초기화: 수집을 중단했습니다.',handNames{side});
                end
                firstPlotMs(side) = NaN;
                lastPlotMs(side) = NaN;
                for cc = 1:3
                    for ll = curves{side,cc}, clearpoints(ll); end
                end
            end
            lastDeviceMs(side) = frame.time_ms;
            updatePlot(side,frame);
            if running, storeFrame(side,frame); end
        catch err
            % Never silently continue a recording after an unhandled error.
            if running
                running = false;
                message = sprintf('%s 처리 오류로 수집 중지: %s',handNames{side},err.message);
                flushBuffers();
            end
            warning('XIAO:BLEReceive', '%s BLE 수신 오류: %s',handNames{side},err.message);
        end
    end

    function updatePlot(side,frame)
        ms = frame.time_ms;
        if isnan(firstPlotMs(side)), firstPlotMs(side) = ms; end
        if ~isnan(lastPlotMs(side)) && ms-lastPlotMs(side) < 250
            return;
        end
        t = (ms-firstPlotMs(side))/1000;
        for k = 1:6, addpoints(curves{side,1}(k),t,frame.fsr(k)); end
        for k = 1:3
            addpoints(curves{side,2}(k),t,frame.imu(k));
            addpoints(curves{side,3}(k),t,frame.imu(k+3));
        end
        for cc = 1:3
            xlim(axesHandles(side,cc),[max(0,t-windowSeconds),max(windowSeconds,t)]);
        end
        lastPlotMs(side) = ms;
    end

    function startCapture(~,~)
        if running || sending || submitted || completed, return; end
        for side = 1:2
            if isempty(lastSeen{side}) || toc(lastSeen{side}) > 3
                message = sprintf('%s 센서값이 최근 3초간 수신되지 않았습니다.',handNames{side});
                refreshUI(); return;
            end
        end
        if ~hasRun
            label = strtrim(get(gestureInput,'String'));
            subjectIndex = get(subjectInput,'Value');
            if isempty(label) || subjectIndex == 1 || ...
                    ~isempty(regexp(label,'[0-9]$|[<>:"/\\|?*]','once'))
                message = 'Subject와 수어명을 입력하세요. 수어명은 숫자로 끝날 수 없습니다.';
                refreshUI(); return;
            end
            answer = questdlg(sprintf('%s / %s / 양손 / 20회 측정이 맞습니까?', ...
                subjectOptions{subjectIndex},label),'측정자 확인', ...
                '맞습니다','취소','취소');
            if ~strcmp(answer,'맞습니다'), return; end
            subjectId = sprintf('subject%d',subjectIndex-1);
            gestureName = label;
            stamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
            sessionId = sprintf('s%s_%09d',stamp,randi(999999999));
            if ~isfolder(saveDirectory), mkdir(saveDirectory); end
            safeLabel = regexprep(label,'[<>:"/\\|?*\x00-\x1F]','_');
            for sd = 1:2
                filePaths{sd} = fullfile(saveDirectory,sprintf('%s_%s_%s_%s.csv', ...
                    subjectId,safeLabel,hands{sd},sessionId));
                if isfile(filePaths{sd})
                    error('XIAO:FileExists','CSV가 이미 존재합니다: %s',filePaths{sd});
                end
                write_sensor_csv(filePaths{sd},{},'replace');
            end
            hasRun = true;
            samples = {cell(1,0),cell(1,0)};
            buffers = {cell(1,0),cell(1,0)};
            pending = cell(1,0);
            sequence = zeros(1,2);
            trialCounts = zeros(2,20);
            sent = 0;
            elapsedMs = 0;
            elapsedOffsetMs = 0;
            lastFlush = tic;
        end
        clock = tic;
        running = true;
        message = '양손 동시 수집 중입니다 (5초 측정 + 3초 휴식).';
        refreshUI();
    end

    function storeFrame(side,frame)
        elapsedMs = elapsedOffsetMs+toc(clock)*1000; % Shared host clock.
        phase = sensor_collection_phase(elapsedMs);
        if phase.done
            finishCapture(); return;
        end
        if ~phase.collecting, return; end
        sequence(side) = sequence(side)+1;
        trialId = sprintf('%s%d',gestureName,phase.trial);
        payload = make_sensor_payload(frame.fsr,frame.imu(1:6),hands{side}, ...
            sequence(side),frame.time_ms,trialId,subjectId,sessionId);
        samples{side}{end+1} = payload;
        buffers{side}{end+1} = payload;
        pending{end+1} = payload;
        trialCounts(side,phase.trial) = trialCounts(side,phase.trial)+1;
        if numel(buffers{side}) >= 20, flushBuffers(); end
    end

    function stopCapture(~,~)
        if ~running, return; end
        elapsedMs = elapsedOffsetMs+toc(clock)*1000;
        elapsedOffsetMs = elapsedMs;
        running = false;
        flushBuffers();
        message = sprintf('일시 중지 (%.1f초 진행). 계속 수집할 수 있습니다.',elapsedMs/1000);
        refreshUI();
    end

    function finishCapture()
        if ~running, return; end
        elapsedMs = 160000;
        running = false;
        completed = true;
        flushBuffers();
        if all(trialCounts(:) > 0)
            message = '양손 20회 수집 완료. 결과 확인 후 전송하세요.';
        else
            message = '20회 측정 종료. 일부 회차에 데이터가 없어 CSV 확인이 필요합니다.';
        end
    end

    function flushBuffers()
        for sd = 1:2
            if isempty(buffers{sd}) || isempty(filePaths{sd}), continue; end
            try
                write_sensor_csv(filePaths{sd},buffers{sd},'append');
                buffers{sd} = cell(1,0);
            catch err
                running = false;
                message = sprintf('%s CSV 저장 실패: %s',handNames{sd},err.message);
            end
        end
        lastFlush = tic;
    end

    function deleteLastTrial(~,~)
        if running || sending || submitted || ~hasRun, return; end
        trial = find(any(trialCounts > 0,1),1,'last');
        if isempty(trial), return; end
        answer = questdlg(sprintf('%d회차의 양손 데이터를 삭제하시겠습니까?',trial), ...
            '회차 삭제 확인','삭제','취소','취소');
        if ~strcmp(answer,'삭제'), return; end
        id = sprintf('%s%d',gestureName,trial);
        for sd = 1:2
            keep = ~cellfun(@(p) strcmp(p.source,id),samples{sd});
            write_sensor_csv(filePaths{sd},samples{sd}(keep),'replace');
            samples{sd} = samples{sd}(keep);
            buffers{sd} = cell(1,0);
        end
        pending = pending(~cellfun(@(p) strcmp(p.source,id),pending));
        trialCounts(:,trial) = 0;
        elapsedOffsetMs = (trial-1)*8000;
        elapsedMs = elapsedOffsetMs;
        completed = false;
        message = sprintf('%d회차 양손 삭제. 계속 수집을 누르면 해당 회차부터 다시 측정합니다.',trial);
        refreshUI();
    end

    function correctMetadata(~,~)
        if running || sending || submitted || ~hasRun, return; end
        label = strtrim(get(gestureInput,'String'));
        subjectIndex = get(subjectInput,'Value');
        if isempty(label) || subjectIndex == 1 || ...
                ~isempty(regexp(label,'[0-9]$|[<>:"/\\|?*]','once'))
            message = '올바른 Subject와 수어명을 선택하세요.'; refreshUI(); return;
        end
        newSubject = sprintf('subject%d',subjectIndex-1);
        if strcmp(label,gestureName) && strcmp(newSubject,subjectId), return; end
        answer = questdlg(sprintf('수집된 양손 데이터의 정보를 %s / %s 로 변경합니까?', ...
            newSubject,label),'정보 수정 확인','수정','취소','취소');
        if ~strcmp(answer,'수정'), return; end
        originalName = gestureName;
        for sd = 1:2
            for k = 1:numel(samples{sd})
                p = samples{sd}{k};
                trialNumber = str2double(extractAfter(string(p.source),strlength(originalName)));
                p.source = sprintf('%s%d',label,trialNumber);
                p.subject_id = newSubject;
                samples{sd}{k} = p;
            end
            write_sensor_csv(filePaths{sd},samples{sd},'replace');
            buffers{sd} = cell(1,0);
        end
        for k = 1:numel(pending)
            trialNumber = str2double(extractAfter(string(pending{k}.source),strlength(originalName)));
            pending{k}.source = sprintf('%s%d',label,trialNumber);
            pending{k}.subject_id = newSubject;
        end
        gestureName = label;
        subjectId = newSubject;
        message = '양손 CSV 및 전송 대기 데이터의 Subject/수어명을 수정했습니다.';
        refreshUI();
    end

    function resetCapture(~,~)
        if running || sending || ~isempty(transfer), return; end
        if hasRun
            answer = questdlg(['새 수집을 시작하시겠습니까?' newline ...
                '기존 양손 CSV 파일은 삭제하지 않고 유지합니다.'], ...
                '새 수집 확인','새 수집','취소','취소');
            if ~strcmp(answer,'새 수집'), return; end
            flushBuffers();
        end
        hasRun = false;
        submitted = false;
        completed = false;
        running = false;
        elapsedMs = 0;
        elapsedOffsetMs = 0;
        clock = [];
        gestureName = '';
        subjectId = '';
        sessionId = '';
        filePaths = {'',''};
        samples = {cell(1,0),cell(1,0)};
        buffers = {cell(1,0),cell(1,0)};
        pending = cell(1,0);
        trialCounts = zeros(2,20);
        sequence = zeros(1,2);
        sent = 0;
        message = '새 수집 준비 완료. 이전 CSV는 sensor_data 폴더에 보존됩니다.';
        refreshUI();
    end

    function sendCapture(~,~)
        if running || sending || submitted || isempty(pending), return; end
        flushBuffers();
        answer = questdlg(sprintf('양손 %d개 데이터를 Python 서버로 전송할까요?\n서버를 먼저 실행해야 합니다.', ...
            numel(pending)), '전송 확인', '전송','취소','취소');
        if ~strcmp(answer,'전송'), return; end
        submitted = true;
        sending = true;
        message = 'Python 서버로 전송 중입니다. 중복 방지를 위해 실패 시 자동 재전송하지 않습니다.';
        refreshUI();
    end

    function progressTransfer()
        if ~isempty(transfer) && strcmp(transfer.State,'finished')
            try
                result = fetchOutputs(transfer);
            catch err
                result = struct('sent',0,'unconfirmed',numel(activeBatch),'message',err.message);
            end
            delete(transfer);
            transfer = [];
            sent = sent+result.sent;
            if result.unconfirmed > 0
                pending = [activeBatch(result.sent+1:end) pending];
                sending = false;
                message = ['HTTP 오류, CSV는 유지됩니다: ' result.message];
            end
            activeBatch = cell(1,0);
        end
        if sending && isempty(transfer) && ~isempty(pending)
            n = min(20,numel(pending));
            activeBatch = pending(1:n);
            pending(1:n) = [];
            try
                transfer = parfeval(backgroundPool,@post_sensor_batch,1,endpoint,activeBatch);
            catch err
                pending = [activeBatch pending];
                activeBatch = cell(1,0);
                sending = false;
                message = ['HTTP 작업 시작 실패: ' err.message];
            end
        end
        if sending && isempty(transfer) && isempty(pending)
            sending = false;
            message = sprintf('Python 전송 완료: %d개 확인.',sent);
        end
    end

    function refreshUI()
        if ~isgraphics(fig), return; end
        if running && elapsedOffsetMs+toc(clock)*1000 >= 160000
            finishCapture();
        end
        if toc(lastFlush) >= 1, flushBuffers(); end
        progressTransfer();
        connectionState = cell(1,2);
        for sd = 1:2
            if isempty(lastSeen{sd})
                connectionState{sd} = '데이터 대기';
            elseif toc(lastSeen{sd}) > 3
                connectionState{sd} = '수신 끊김?';
            else
                connectionState{sd} = sprintf('수신 중 (누락추정 %d)',missingFrames(sd));
            end
        end
        if running
            elapsedMs = elapsedOffsetMs+toc(clock)*1000;
            phase = sensor_collection_phase(min(elapsedMs,160000));
            if phase.collecting, action = '측정'; else, action = '휴식'; end
            stage = sprintf('%s %d/20회 | 남은 %.1f초',action,phase.trial,phase.remainingSec);
        else
            stage = message;
        end
        statusLabel.String = sprintf('%s | L:%s(%d) R:%s(%d) | 전송대기:%d, 확인:%d', ...
            stage,connectionState{1},received(1),connectionState{2},received(2), ...
            numel(pending),sent);
        editable = ~running && ~sending && ~submitted;
        enable(startButton, editable && ~completed);
        enable(stopButton,running);
        enable(sendButton,~running && ~sending && ~submitted && ~isempty(pending));
        enable(deleteButton,editable && hasRun && any(trialCounts(:)>0));
        enable(correctButton,editable && hasRun);
        enable(clearButton,~running && ~sending);
        enable(gestureInput,editable);
        enable(subjectInput,editable);
        if hasRun, startButton.String = '계속 수집';
        else, startButton.String = '양손 20회 수집 시작'; end
    end

    function changeFsrYMax(~,~)
        selectedMax = fsrYMaxValues(get(fsrYMaxPopup,'Value'));
        for hand = 1:2
            if isgraphics(axesHandles(hand,1))
                ylim(axesHandles(hand,1),[0 selectedMax]);
            end
        end
        drawnow limitrate;
    end

    function cleanupConnection()
        if running, flushBuffers(); end
        if ~isempty(filePaths{1}) || ~isempty(filePaths{2})
            % Closing the plot should not silently lose buffered samples.
            flushBuffers();
        end
        if ~isempty(transfer)
            try, cancel(transfer); delete(transfer); catch, end
            transfer = [];
        end
        for sd = 1:2
            if ~isempty(chars{sd})
                try
                    chars{sd}.DataAvailableFcn = [];
                    unsubscribe(chars{sd});
                catch
                end
                chars{sd} = [];
            end
            devices{sd} = [];
        end
        fprintf('[XIAO BLE] 연결 종료. 기존 저장된 CSV는 유지됩니다.\n');
    end
end

function enable(control,allowed)
if allowed, control.Enable = 'on'; else, control.Enable = 'off'; end
end
