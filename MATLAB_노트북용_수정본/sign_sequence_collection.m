function collector = sign_sequence_collection(saveDirectory, timeProvider)
% Fixed-window experiment companion; receives frames from the existing monitor.
% New files only. Each step has its own server-compatible session/CSV; the
% manifest records BOTH steps on one unpaused experiment clock.
% Optional timeProvider returns seconds (deterministic tests only).
if nargin<1, saveDirectory=fullfile(fileparts(mfilename('fullpath')),'sensor_data','sequences'); end
dataKind="synthetic";
if nargin<2, origin=tic; timeProvider=@() toc(origin); dataKind="real"; end
fig=figure('Name','두 수어 조합 실험 (고정 구간)','NumberTitle','off', ...
    'Position',[100 100 690 310],'Tag','SignSequenceCollection','CloseRequestFcn',@closeFigure);
uicontrol(fig,'Style','text','String','Subject / 첫 수어 / 두 번째 수어 / 조합 반복 수', ...
    'Position',[20 265 640 25]);
subject=uicontrol(fig,'Style','popupmenu','String',arrayfun(@(n)sprintf('subject%d',n),1:10,'UniformOutput',false), ...
    'Position',[20 225 140 30],'Tag','SequenceSubject');
first=uicontrol(fig,'Style','edit','String','머리','Position',[175 225 150 30],'Tag','SequenceFirst');
second=uicontrol(fig,'Style','edit','String','아프다','Position',[340 225 150 30],'Tag','SequenceSecond');
count=uicontrol(fig,'Style','edit','String','20','Position',[510 225 100 30],'Tag','SequenceCount');
startButton=uicontrol(fig,'Style','pushbutton','String','새 실험 시작','Position',[20 175 190 35], ...
    'Callback',@startCapture,'Tag','SequenceStart');
uicontrol(fig,'Style','pushbutton','String','중지 (파일 보존)','Position',[230 175 190 35], ...
    'Callback',@stopCapture,'Tag','SequenceStop');
status=uicontrol(fig,'Style','text','String','두 장치 연결 후 시작하세요. 왼손으로 두 수어 수행 / 오른손 내림. 양손 저장, 왼손 모델 평가.', ...
    'Position',[20 65 640 100],'HorizontalAlignment','left','Tag','SequenceStatus');
running=false; started=0; lastSeen=[-Inf -Inf]; lastDevice=[NaN NaN]; lastFlush=-Inf;
plan=table; folder=''; recordId=''; labels={}; n=0;
samples={}; buffers={}; sequence=[]; participant='';
flushing=false; ticking=false;
pulse=timer('ExecutionMode','fixedSpacing','Period',0.1,'BusyMode','drop','TimerFcn',@tick);
start(pulse);
collector=struct('addFrame',@addFrame,'tick',@tick,'close',@closeFigure,'figure',fig);

    function startCapture(~,~)
        if running||flushing||ticking, return; end
        if any(timeProvider()-lastSeen>3)
            status.String='양쪽 장치에서 최근 3초 안에 데이터가 수신돼야 시작할 수 있습니다.'; return;
        end
        labels={strtrim(first.String),strtrim(second.String)};
        n=str2double(count.String);
        if ~isfinite(n)||n<1||n>100||n~=fix(n)||any(cellfun(@isempty,labels))|| ...
                any(cellfun(@(s)~isempty(regexp(s,'[0-9]$|[<>:"/\\|?*]','once')),labels))
            status.String='수어명(숫자로 끝나지 않음)과 반복 수(1~100)를 확인하세요.'; return;
        end
        participant=sprintf('subject%d',subject.Value);
        recordId=['seq_' char(java.util.UUID.randomUUID())];
        folder=fullfile(saveDirectory,recordId); mkdir(folder);
        % Declare all attempts BEFORE recording, so absent/aborted ones cannot
        % disappear from the experiment accounting.
        participant_id=repmat(string(participant),n*2,1);
        attempt_id=strings(n*2,1); step=repmat([1;2],n,1);
        actual_label=strings(n*2,1); csv_path=strings(n*2,1);
        session_id=strings(n*2,1); trial_id=strings(n*2,1);
        hand=repmat("left",n*2,1); recording_id=repmat(string(recordId),n*2,1);
        window_start_ms=(0:n*2-1)'*8000; window_end_ms=window_start_ms+5000;
        captureStatus=repmat("not_started",n*2,1);
        for k=1:n*2
            attempt_id(k)=sprintf('%s_A%03d',recordId,ceil(k/2));
            actual_label(k)=string(labels{step(k)});
            session_id(k)=sprintf('sseq_%s_%03d',strrep(recordId,'-','_'),k);
            trial_id(k)=actual_label(k)+"1";
            csv_path(k)=sprintf('window_%03d_L.csv',k);
        end
        plan=table(participant_id,attempt_id,step,actual_label,csv_path,session_id,trial_id,hand, ...
            recording_id,window_start_ms,window_end_ms,captureStatus, ...
            'VariableNames',{'participant_id','attempt_id','step','actual_label','csv_path', ...
            'session_id','trial_id','hand','recording_id','window_start_ms','window_end_ms','status'});
        samples=cell(n*2,2); buffers=cell(n*2,2); sequence=zeros(n*2,2);
        for k=1:n*2
            for side=1:2
                samples{k,side}={}; buffers{k,side}={};
                write_sensor_csv(rawPath(k,side),{},'replace',3);
            end
        end
        writetable(plan,fullfile(folder,'attempt_manifest.csv'),'Encoding','UTF-8');
        started=timeProvider()+3;
        plan.data_kind=repmat(dataKind,n*2,1);
        plan.capture_start_utc=repmat(string(datetime('now','TimeZone','UTC')+seconds(3), ...
            'yyyy-MM-dd''T''HH:mm:ss.SSSXXX'),n*2,1);
        writetable(plan,fullfile(folder,'attempt_manifest.csv'),'Encoding','UTF-8');
        running=true;
        set([subject first second count startButton],'Enable','off');
        status.String=['3초 뒤 시작. 5초 동작 / 3초 휴식. 저장: ' folder];
    end

    function addFrame(side,frame)
        now=timeProvider(); lastSeen(side)=now;
        if running && ~isnan(lastDevice(side)) && frame.time_ms<lastDevice(side)
            stopWithReason('device_clock_reset');
        end
        lastDevice(side)=frame.time_ms;
        if ~running||now<started, return; end
        phase=sign_sequence_phase((now-started)*1000,n);
        if phase.done, tick(); return; end
        if ~phase.collecting, return; end
        k=(phase.attempt-1)*2+phase.step;
        try
        hands='LR'; sequence(k,side)=sequence(k,side)+1;
        payload=make_sensor_payload(frame.fsr,frame.imu(1:6),hands(side),sequence(k,side), ...
            frame.time_ms,char(plan.trial_id(k)),participant,char(plan.session_id(k)), ...
            'left_only',phase.windowMs,phase.windowMs);
        samples{k,side}{end+1}=payload; buffers{k,side}{end+1}=payload;
        if numel(buffers{k,side})>=20, flush(); end
        catch err
            try
                stopWithReason('capture_error');
            catch saveError
                running=false;
                status.String=['수집/저장 오류. 파일 확인 필요: ' saveError.message];
            end
            rethrow(err);
        end
    end

    function tick(varargin)
        if ~running||~isgraphics(fig)||ticking, return; end
        ticking=true;
        elapsed=(timeProvider()-started)*1000;
        if elapsed<0
            status.String=sprintf('준비 %.1f초',-elapsed/1000); ticking=false; return;
        end
        try
            ended=plan.window_end_ms<=elapsed & plan.status~="complete";
            plan.status(ended)="complete"; % Sensor completeness is evaluated later.
            if any(ended)||timeProvider()-lastFlush>=1
                flush();lastFlush=timeProvider();
            end
            if any(ended)
                writetable(plan,fullfile(folder,'attempt_manifest.csv'),'Encoding','UTF-8');
            end
            phase=sign_sequence_phase(elapsed,n);
            if phase.done
                running=false; startButton.Enable='on';
                set([subject first second count],'Enable','on');
                status.String=['수집 종료. attempt_manifest.csv를 Python으로 평가하세요. ' folder];
            else
                action='휴식'; if phase.collecting, action=labels{phase.step}; end
                status.String=sprintf('조합 %d/%d, 동작 %d: %s (%.1f초)\n%s', ...
                    phase.attempt,n,phase.step,action,phase.windowMs/1000,folder);
            end
        catch err
            running=false; status.String=['저장 오류. 새 측정 대신 파일을 먼저 확인하세요: ' err.message];
        end
        ticking=false;
    end

    function flush()
        if flushing,return;end
        flushing=true;
        try
        for k=1:size(buffers,1)
            for side=1:2
                if isempty(buffers{k,side}), continue; end
                % Detach before IO: timer/BLE callbacks may run during writes.
                % New arrivals belong to the next batch, never this one.
                batch=buffers{k,side};
                buffers{k,side}={};
                try
                    write_sensor_csv(rawPath(k,side),batch,'append');
                catch err
                    buffers{k,side}=[batch buffers{k,side}];
                    rethrow(err);
                end
            end
        end
        catch err
            flushing=false;rethrow(err);
        end
        flushing=false;
    end

    function path=rawPath(k,side)
        hands='LR'; path=fullfile(folder,sprintf('window_%03d_%s.csv',k,hands(side)));
    end

    function stopCapture(varargin)
        stopWithReason('interrupted');
    end

    function stopWithReason(reason)
        if ~running, return; end
        tick(); % Finalize windows that finished before pressing stop.
        running=false;
        elapsed=(timeProvider()-started)*1000;
        interrupted=plan.window_start_ms<=elapsed & plan.status~="complete";
        plan.status(interrupted)=string(reason);
        flush(); writetable(plan,fullfile(folder,'attempt_manifest.csv'),'Encoding','UTF-8');
        set([subject first second count startButton],'Enable','on');
        status.String=['중지. 미완료 시도는 평가에서 제외하며 파일은 보존합니다. ' folder];
    end

    function closeFigure(varargin)
        stopCapture();
        if isvalid(pulse), stop(pulse); delete(pulse); end
        if isgraphics(fig), delete(fig); end
    end
end
