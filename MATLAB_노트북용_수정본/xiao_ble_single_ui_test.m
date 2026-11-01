function xiao_ble_single_ui_test(c)
% XIAO_BLE_SINGLE_UI_TEST  Preview the two-hand layout with one real BLE board.
% Usage after BLE connection and subscription:
%   xiao_ble_single_ui_test(c)
% Capture 20 trials, persist CSV, then upload with the explicit Send button.
% Requires xiao_ble_packet_decode.m in MATLAB Current Folder / path.

if nargin ~= 1
    error('XIAO:Input', '이미 연결한 BLE characteristic c를 입력하세요: xiao_ble_single_ui_test(c)');
end
if exist('xiao_ble_packet_decode','file') ~= 2
    error('XIAO:Decoder', 'xiao_ble_packet_decode.m 파일이 MATLAB 경로에 없습니다.');
end

required = {'make_sensor_payload','write_sensor_csv','post_sensor_batch','sensor_collection_phase'};
for k=1:numel(required)
    if exist(required{k},'file') ~= 2
        error('XIAO:MissingFile','%s.m 파일을 MATLAB 경로에 추가하세요.',required{k});
    end
end
helpers=bindSensorHelpers();
endpoint = 'http://100.92.15.45:5000/sensor';
saveDirectory = fullfile(fileparts(mfilename('fullpath')),'sensor_data');
running=false; sending=false; submitted=false; completed=false;
samples={}; unsaved={}; pending={}; activeBatch={}; transfer=[];
elapsedOffsetMs=0; captureClock=[]; lastSeen=[]; lastDeviceMs=NaN;
subjectId=''; gestureName=''; sessionId=''; csvPath=''; deviceId='L';
message='수신 확인 후 측정자·수어명·손을 선택하세요.';
latestSample=[];
discarding=false;
clockTimer=[];
% The current c should already be subscribed, and old callbacks can conflict.
try
    c.DataAvailableFcn = [];
catch err
    error('XIAO:Characteristic','BLE characteristic을 확인할 수 없습니다: %s',err.message);
end

f = figure('Name','XIAO BLE 양손 센서 모니터 (1개 보드 실시간 테스트)', ...
    'NumberTitle','off','Color','w','Position',[70 70 1300 860], ...
    'CloseRequestFcn',@closeWindow);

uicontrol(f,'Style','text','String','수어명','Units','normalized', ...
    'Position',[0.02 0.955 0.06 0.03],'BackgroundColor','w');
gestureInput=uicontrol(f,'Style','edit','String','','Units','normalized', ...
    'Position',[0.085 0.95 0.15 0.045]);
subjectInput=uicontrol(f,'Style','popupmenu','String',{'측정자 선택', 'subject1 (정성윤)', ...
    'subject2 (임영훈)', 'subject3 (주다경)', 'subject4 (고준혁)', ...
    'subject5 (지상민)', 'subject6 (이수민)', 'subject7 (박경민)', ...
    'subject8 (백수현)', 'subject9 (이하영)', 'subject10 (정소윤)'}, ...
    'Units','normalized','Position',[0.25 0.95 0.22 0.045]);
startButton=uicontrol(f,'Style','pushbutton','String','20회 수집 시작', ...
    'Units','normalized','Position',[0.49 0.95 0.16 0.045],'Callback',@startCapture);
pauseButton=uicontrol(f,'Style','pushbutton','String','일시 중지', ...
    'Units','normalized','Position',[0.67 0.95 0.10 0.045],'Callback',@pauseCapture);
resetButton=uicontrol(f,'Style','pushbutton','String','새 수집', ...
    'Units','normalized','Position',[0.79 0.95 0.10 0.045],'Callback',@resetCapture);
handInput=uicontrol(f,'Style','popupmenu','String',{'왼손','오른손'}, ...
    'Units','normalized','Position',[0.90 0.95 0.09 0.045]);
sendButton=uicontrol(f,'Style','pushbutton','String','Python 서버로 전송', ...
    'Units','normalized','Position',[0.02 0.895 0.18 0.045],'Enable','off','Callback',@sendCapture);
discardButton=uicontrol(f,'Style','pushbutton','String','중지하고 버리기', ...
    'Units','normalized','Position',[0.21 0.895 0.16 0.045], ...
    'Enable','off','Callback',@discardCapture);
uicontrol(f,'Style','text','String',endpoint,'Units','normalized', ...
    'Position',[0.38 0.895 0.36 0.04],'BackgroundColor','w','HorizontalAlignment','left');
% FSR ADC Y-axis scaling: shared by both FSR charts, display only.
fsrYMaxValues = [50 100 200 500 1000 4095];
uicontrol(f,'Style','text','String','Y max (ADC)', ...
    'Units','normalized','Position',[0.75 0.895 0.11 0.04], ...
    'BackgroundColor','w','HorizontalAlignment','right');
fsrYMaxPopup = uicontrol(f,'Style','popupmenu', ...
    'String',cellstr(string(fsrYMaxValues)), ...
    'Value',numel(fsrYMaxValues),'Units','normalized', ...
    'Position',[0.87 0.895 0.11 0.045], ...
    'Callback',@changeFsrYMax);
status = uicontrol(f,'Style','text','String', ...
    '단일 보드 BLE 수집 / CSV 저장 / 수집 후 서버 전송', ...
    'Units','normalized','Position',[0.02 0.845 0.96 0.035], ...
    'BackgroundColor','w','HorizontalAlignment','left');
panel = uipanel(f,'BorderType','none','BackgroundColor','w', ...
    'Units','normalized','Position',[0 0.015 1 0.82]);
layout = tiledlayout(panel,2,3,'TileSpacing','compact','Padding','compact');
colors = lines(6);
plots = cell(2,3);
axs = gobjects(2,3);
handNames={'왼손','오른손'};
for side=1:2
    for j=1:3
        ax=nexttile(layout,(side-1)*3+j);
        axs(side,j)=ax;
        hold(ax,'on');grid(ax,'on');xlabel(ax,'Time (s)');xlim(ax,[0 20]);
        if j==1
            title(ax,sprintf('%s FSR (A0~A5)',handNames{side}));
            ylabel(ax,'ADC raw');ylim(ax,[0 fsrYMaxValues(end)]);
            legendNames=arrayfun(@(k)sprintf('FSR%d',k),1:6,'UniformOutput',false);
        elseif j==2
            title(ax,sprintf('%s IMU 가속도',handNames{side}));
            ylabel(ax,'g');legendNames={'AX','AY','AZ'};
        else
            title(ax,sprintf('%s IMU 자이로',handNames{side}));
            ylabel(ax,'deg/s');legendNames={'GX','GY','GZ'};
        end
        handles=gobjects(1,numel(legendNames));
        for k=1:numel(legendNames)
            handles(k)=animatedline(ax,'Color',colors(k,:), ...
                'LineWidth',1.2,'MaximumNumPoints',300);
        end
        plots{side,j}=handles;
        legend(ax,handles,legendNames,'Location','northeast', ...
            'AutoUpdate','off','FontSize',7);
    end
end

buffer=[];
firstMs=NaN;
lastDrawMs=NaN;
frames=0;
errors=0;
try
    % Notification callback keeps the figure responsive. Existing subscription
    % stays active; do not open a second BLE connection.
    c.DataAvailableFcn=@onNotification;
    clockTimer=timer('ExecutionMode','fixedSpacing','Period',0.25, ...
        'BusyMode','drop','TimerFcn',@tick);
    start(clockTimer);
    refreshUI();
catch err
    closeWindow([],[]);
    rethrow(err)
end

    function onNotification(src,~)
        if ~isgraphics(f),return;end
        try
            bytes=read(src,'oldest');
            [buffer,sample]=xiao_ble_packet_decode(buffer,bytes);
            if isempty(sample),return;end
            lastSeen=tic;
            if ~isnan(lastDeviceMs) && double(sample.time_ms)<lastDeviceMs
                if running, pauseCapture([],[]); end
                completed=true;
                message='보드 재시작 감지: 새 수집으로 시작하세요. 기존 CSV는 보존됩니다.';
                firstMs=NaN; lastDrawMs=NaN;
                for sd=1:2
                    for col=1:3
                        for line=plots{sd,col}, clearpoints(line); end
                    end
                end
            end
            lastDeviceMs=double(sample.time_ms);
            frames=frames+1;
            if running, storeFrame(sample); end
            latestSample=sample;
        catch err
            errors=errors+1;
            if running, pauseCapture([],[]); end
            message=['데이터 처리 오류: ' err.message];
            if errors<=3
                warning('XIAO:SinglePreview','BLE 데이터 처리 오류: %s',err.message);
            end
        end
    end

    function drawLatest()
            if isempty(latestSample), return; end
            sample=latestSample;
            if isnan(firstMs),firstMs=double(sample.time_ms);end
            ms=double(sample.time_ms);
            if ~isnan(lastDrawMs) && ms-lastDrawMs<200,return;end
            lastDrawMs=ms;
            t=max(0,(ms-firstMs)/1000);
            side=get(handInput,'Value');
            for k=1:6
                addpoints(plots{side,1}(k),t,double(sample.fsr(k)));
            end
            for k=1:3
                addpoints(plots{side,2}(k),t,double(sample.imu(k)));
                addpoints(plots{side,3}(k),t,double(sample.imu(k+3)));
            end
            for j=1:3
                xlim(axs(side,j),[max(0,t-20),max(20,t)]);
            end
    end

    function changeFsrYMax(~,~)
        selectedMax = fsrYMaxValues(get(fsrYMaxPopup,'Value'));
        for hand = 1:2
            if isgraphics(axs(hand,1))
                ylim(axs(hand,1),[0 selectedMax]);
            end
        end
        drawnow limitrate;
    end

    function startCapture(~,~)
        if running || sending || submitted || completed, return; end
        if isempty(lastSeen) || toc(lastSeen)>3
            message='최근 3초간 센서 수신이 없습니다. BLE를 재연결하세요.';
            refreshUI(); return;
        end
        if isempty(sessionId)
            label=strtrim(get(gestureInput,'String'));
            subjectIndex=get(subjectInput,'Value');
            if isempty(label) || subjectIndex==1 || ...
                    ~isempty(regexp(label,'[0-9]$|[<>:"/\\|?*\x00-\x1F]','once'))
                message='측정자와 수어명을 입력하세요. 수어명 끝에는 숫자를 쓰지 마세요.';
                refreshUI(); return;
            end
            if get(handInput,'Value')==1, deviceId='L'; else, deviceId='R'; end
            subjectId=sprintf('subject%d',subjectIndex-1);
            gestureName=label;
            sessionId=['s' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')) ...
                sprintf('_%09d',randi(999999999))];
            try
                if ~isfolder(saveDirectory), mkdir(saveDirectory); end
                csvPath=fullfile(saveDirectory,sprintf('%s_%s_%s.csv',subjectId,deviceId,sessionId));
                if isfile(csvPath)
                    error('XIAO:FileExists','이미 존재하는 CSV입니다: %s',csvPath);
                end
            catch err
                sessionId=''; message=['CSV 생성 실패: ' err.message]; refreshUI(); return;
            end
        end
        captureClock=tic; running=true;
        message='5초 측정 + 3초 휴식, 20회 수집 중'; refreshUI();
    end

    function storeFrame(sample)
        phase=helpers.phase(elapsedOffsetMs+toc(captureClock)*1000);
        if phase.done, finishCapture(); return; end
        if ~phase.collecting, return; end
        payload=helpers.makePayload(sample.fsr,sample.imu(1:6),deviceId, ...
            numel(samples)+1,double(sample.time_ms), ...
            sprintf('%s%d',gestureName,phase.trial),subjectId,sessionId);
        samples{end+1}=payload;
        unsaved{end+1}=payload;
        pending{end+1}=payload;
    end

    function pauseCapture(~,~)
        if ~running, return; end
        elapsedOffsetMs=elapsedOffsetMs+toc(captureClock)*1000;
        running=false;
        message='일시 중지: 계속 수집하거나 현재 데이터를 전송할 수 있습니다.';
        flushCsv(); refreshUI();
    end

    function finishCapture()
        elapsedOffsetMs=160000; running=false; completed=true;
        message='20회 측정 종료. CSV 확인 후 전송하세요.';
        flushCsv();
    end

    function ok=flushCsv()
        ok=true;
        if isempty(unsaved), return; end
        try
            % Atomic full snapshot avoids duplicating rows after a failed append.
            helpers.writeCsv(csvPath,samples,'replace');
            unsaved={};
        catch err
            ok=false;
            if running
                elapsedOffsetMs=elapsedOffsetMs+toc(captureClock)*1000;
            end
            running=false;
            message=['CSV 저장 실패, 전송 차단: ' err.message];
        end
    end

    function resetCapture(~,~)
        if running || sending || discarding, return; end
        if ~isempty(samples)
            if ~strcmp(questdlg('기존 CSV를 보존하고 새 수집을 시작할까요?', ...
                    '새 수집','새 수집','취소','취소'),'새 수집'), return; end
            if ~flushCsv(), refreshUI(); return; end
        end
        clearSession();
        message='새 수집 준비. 기존 CSV는 보존됩니다.'; refreshUI();
    end

    function discardCapture(~,~)
        if sending || submitted || isempty(sessionId) || discarding, return; end
        % Stop before showing a dialog. Never save the unwanted session here.
        if running, elapsedOffsetMs=elapsedOffsetMs+toc(captureClock)*1000; end
        running=false; discarding=true;
        message='수집 중지. 이번 세션 폐기 여부를 확인하세요.'; refreshUI();
        answer=questdlg('이번 세션 전체를 버릴까요? 이미 저장한 이번 CSV도 삭제합니다. 이전 세션은 유지됩니다.', ...
            '수집 중지','버리기','유지','유지');
        discarding=false;
        if ~strcmp(answer,'버리기')
            message='일시 중지 상태로 유지합니다. 계속 수집하거나 저장·전송할 수 있습니다.';
            refreshUI(); return;
        end
        try
            if isfile(csvPath), delete(csvPath); end
        catch err
            message=['이번 CSV 삭제 실패. 데이터는 유지됩니다: ' err.message];
            refreshUI(); return;
        end
        clearSession();
        message='이번 수집을 버렸습니다. CSV 저장·서버 전송 없이 새 수집할 수 있습니다.';
        refreshUI();
    end

    function clearSession()
        samples={}; unsaved={}; pending={}; activeBatch={};
        sessionId=''; csvPath=''; elapsedOffsetMs=0;
        submitted=false; completed=false; running=false;
    end

    function sendCapture(~,~)
        if running || sending || submitted || discarding || isempty(pending), return; end
        if ~flushCsv(), refreshUI(); return; end
        answer=questdlg(sprintf('%d개 센서값을 서버로 전송할까요?\n%s', ...
            numel(pending),endpoint),'서버 전송','전송','취소','취소');
        if ~strcmp(answer,'전송'), return; end
        submitted=true; sending=true; completed=true;
        message='CSV 저장 완료. 서버 전송 중...';
        refreshUI();
    end

    function tick(~,~)
        if ~isgraphics(f), return; end
        try
            if running
                if isempty(lastSeen) || toc(lastSeen)>3
                    pauseCapture([],[]);
                    message='BLE 수신 중단: 수집을 일시 중지했습니다.';
                elseif elapsedOffsetMs+toc(captureClock)*1000>=160000
                    finishCapture();
                end
            end
            % Keep acquisition in memory. Persist only on pause, finish, send,
            % reset or close, so disk I/O cannot block Stop during capture.
            if ~isempty(transfer) && strcmp(transfer.State,'finished')
                try
                    result=fetchOutputs(transfer);
                catch err
                    result=struct('sent',0,'unconfirmed',numel(activeBatch),'message',err.message);
                end
                delete(transfer); transfer=[];
                pending=[activeBatch(result.sent+1:end) pending];
                activeBatch={};
                if result.unconfirmed>0
                    sending=false;
                    message=sprintf('전송 확인 %d / 미확인 %d. 자동 재시도 없음. CSV: %s | %s', ...
                        numel(samples)-numel(pending),numel(pending),csvPath,result.message);
                end
            end
            if sending && isempty(transfer)
                if isempty(pending)
                    sending=false;
                    message=sprintf('서버 전송 완료: %d개 / CSV: %s',numel(samples),csvPath);
                else
                    n=min(20,numel(pending)); activeBatch=pending(1:n); pending(1:n)=[];
                    try
                        transfer=parfeval(backgroundPool,helpers.postBatch,1,endpoint,activeBatch);
                    catch err
                        pending=[activeBatch pending]; activeBatch={}; sending=false;
                        message=['전송 시작 실패. 자동 재시도 없음. CSV 보존: ' err.message];
                    end
                end
            end
            drawLatest();
            refreshUI();
            drawnow limitrate;
        catch err
            warning('XIAO:MonitorTimer','모니터 처리 오류: %s',err.message);
            message=['모니터 처리 오류: ' err.message];
            refreshUI();
        end
    end

    function refreshUI()
        if ~isgraphics(f), return; end
        locked=~isempty(sessionId);
        set(gestureInput,'Enable',enabled(~locked));
        set(subjectInput,'Enable',enabled(~locked));
        set(handInput,'Enable',enabled(~locked));
        set(startButton,'Enable',enabled(~running && ~sending && ~submitted && ~completed && ~discarding));
        set(pauseButton,'Enable',enabled(running));
        set(resetButton,'Enable',enabled(~running && ~sending && ~discarding));
        set(sendButton,'Enable',enabled(~running && ~sending && ~submitted && ~discarding && ~isempty(pending)));
        set(discardButton,'Enable',enabled(~sending && ~submitted && ~discarding && ~isempty(sessionId)));
        if ~isempty(sessionId) && ~running, set(startButton,'String','계속 수집'); end
        if isempty(sessionId), set(startButton,'String','20회 수집 시작'); end
        progress='';
        if running
            phase=helpers.phase(elapsedOffsetMs+toc(captureClock)*1000);
            if phase.collecting, mode='측정'; else, mode='휴식'; end
            progress=sprintf(' | %d/20 %s %.1f초',phase.trial,mode,phase.remainingSec);
        end
        set(status,'String',sprintf('BLE %d회 | 수집 %d개%s | %s', ...
            frames,numel(samples),progress,message));
    end

    function value=enabled(flag)
        if flag, value='on'; else, value='off'; end
    end

    function closeWindow(~,~)
        if discarding, return; end
        if sending
            if ~strcmp(questdlg('전송 중 닫으면 일부 결과가 미확인될 수 있습니다. 닫을까요?', ...
                    '전송 중','닫기','취소','취소'),'닫기'), return; end
        end
        if ~flushCsv()
            errordlg('CSV 저장 실패: 창을 유지합니다. 경로를 확인하고 다시 시도하세요.');
            return;
        end
        try,c.DataAvailableFcn=[];catch,end
        if ~isempty(clockTimer) && isvalid(clockTimer), stop(clockTimer); delete(clockTimer); end
        if ~isempty(transfer), cancel(transfer); delete(transfer); end
        if isgraphics(f),delete(f);end
    end
end

function helpers=bindSensorHelpers()
% MATLAB resolves Current Folder before addpath. Bind this monitor's helpers
% while its own directory is current, then restore the user's folder.
previousFolder=pwd;
restore=onCleanup(@() cd(previousFolder)); %#ok<NASGU>
cd(fileparts(mfilename('fullpath')));
helpers=struct('makePayload',@make_sensor_payload, ...
    'writeCsv',@write_sensor_csv,'postBatch',@post_sensor_batch, ...
    'phase',@sensor_collection_phase);
end
