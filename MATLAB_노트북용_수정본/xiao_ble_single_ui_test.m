function xiao_ble_single_ui_test(c)
% XIAO_BLE_SINGLE_UI_TEST  Preview the two-hand layout with one real BLE board.
% Usage after BLE connection and subscription:
%   xiao_ble_single_ui_test(c)
% This is visualization only. It never writes CSV or sends HTTP requests.
% Requires xiao_ble_packet_decode.m in MATLAB Current Folder / path.

if nargin ~= 1
    error('XIAO:Input', '이미 연결한 BLE characteristic c를 입력하세요: xiao_ble_single_ui_test(c)');
end
if exist('xiao_ble_packet_decode','file') ~= 2
    error('XIAO:Decoder', 'xiao_ble_packet_decode.m 파일이 MATLAB 경로에 없습니다.');
end

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
uicontrol(f,'Style','edit','String','','Units','normalized', ...
    'Position',[0.085 0.95 0.15 0.045],'Enable','off');
uicontrol(f,'Style','popupmenu','String',{'측정자 선택', 'subject1 (정성윤)', ...
    'subject2 (임영훈)', 'subject3 (주다경)', 'subject4 (고준혁)', ...
    'subject5 (지상민)', 'subject6 (이수민)', 'subject7 (박경민)', ...
    'subject8 (백수현)', 'subject9 (이하영)', 'subject10 (정소윤)'}, ...
    'Units','normalized','Position',[0.25 0.95 0.22 0.045],'Enable','off');
labels = {'양손 20회 수집 시작','일시 중지','정보 수정','새 수집/초기화', ...
          '최근 회차 삭제','Python 서버로 전송'};
positions = {[0.49 0.95 0.16 0.045],[0.67 0.95 0.10 0.045], ...
             [0.79 0.95 0.10 0.045],[0.90 0.95 0.09 0.045], ...
             [0.02 0.895 0.14 0.045],[0.18 0.895 0.17 0.045]};
for n=1:numel(labels)
    uicontrol(f,'Style','pushbutton','String',labels{n},'Units','normalized', ...
        'Position',positions{n},'Enable','off');
end
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
    'XIAO 1개 BLE 실시간 수신 테스트 (CSV 저장 / Python 전송 없음)', ...
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
    set(status,'String','BLE 수신 대기 중... 센서값이 오면 왼손 그래프에 표시됩니다.');
catch err
    if isgraphics(f),delete(f);end
    rethrow(err)
end

    function onNotification(src,~)
        if ~isgraphics(f),return;end
        try
            bytes=read(src,'oldest');
            [buffer,sample]=xiao_ble_packet_decode(buffer,bytes);
            if isempty(sample),return;end
            frames=frames+1;
            if isnan(firstMs),firstMs=double(sample.time_ms);end
            ms=double(sample.time_ms);
            if ~isnan(lastDrawMs) && ms-lastDrawMs<200,return;end
            lastDrawMs=ms;
            t=max(0,(ms-firstMs)/1000);
            for k=1:6
                addpoints(plots{1,1}(k),t,double(sample.fsr(k)));
            end
            for k=1:3
                addpoints(plots{1,2}(k),t,double(sample.imu(k)));
                addpoints(plots{1,3}(k),t,double(sample.imu(k+3)));
            end
            for j=1:3
                xlim(axs(1,j),[max(0,t-20),max(20,t)]);
            end
            set(status,'String',sprintf('XIAO 1개 실시간 수신 중: %d회 | FSR: [%s] | IMU 정상: %d | 오른손은 미연결', ...
                frames,num2str(sample.fsr),double(sample.imu_ok)));
            drawnow limitrate;
        catch err
            errors=errors+1;
            if errors<=3
                warning('XIAO:SinglePreview','BLE 데이터 처리 오류: %s',err.message);
            end
            if isgraphics(status)
                set(status,'String',sprintf('데이터 처리 오류 %d회: %s',errors,err.message));
            end
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

    function closeWindow(~,~)
        try,c.DataAvailableFcn=[];catch,end
        if isgraphics(f),delete(f);end
    end
end
