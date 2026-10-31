function xiao_ble_ui_preview()
% XIAO_BLE_UI_PREVIEW - hardware-free preview of xiao_ble_dual_monitor UI.
% UI layout follows xiao_ble_dual_monitor.m. No BLE, CSV or HTTP operations.

handNames = {'왼손','오른손'};
subjectOptions = {'측정자 선택', 'subject1 (정성윤)', 'subject2 (임영훈)', ...
    'subject3 (주다경)', 'subject4 (고준혁)', 'subject5 (지상민)', ...
    'subject6 (이수민)', 'subject7 (박경민)', 'subject8 (백수현)', ...
    'subject9 (이하영)', 'subject10 (정소윤)'};
fig = figure('Name','XIAO BLE 양손 센서 모니터 (UI 미리보기 - 보드 없음)', ...
    'NumberTitle','off','Color','w','Position',[70 70 1300 860], ...
    'Tag','XIAODualBLEPreview');
uicontrol(fig,'Style','text','String','수어명','Units','normalized', ...
    'Position',[0.02 0.955 0.06 0.03],'BackgroundColor','w');
uicontrol(fig,'Style','edit','String','','Units','normalized', ...
    'Position',[0.085 0.95 0.15 0.045],'BackgroundColor','w');
uicontrol(fig,'Style','popupmenu','String',subjectOptions,'Value',1, ...
    'Units','normalized','Position',[0.25 0.95 0.22 0.045], ...
    'BackgroundColor','w');
% Buttons are intentionally disabled: this file is ONLY a visual preview.
names = {'양손 20회 수집 시작','일시 중지','정보 수정','새 수집/초기화', ...
    '최근 회차 삭제','Python 서버로 전송'};
positions = {[0.49 0.95 0.16 0.045], [0.67 0.95 0.10 0.045], ...
    [0.79 0.95 0.10 0.045], [0.90 0.95 0.09 0.045], ...
    [0.02 0.895 0.14 0.045], [0.18 0.895 0.17 0.045]};
for i = 1:numel(names)
    uicontrol(fig,'Style','pushbutton','String',names{i}, ...
        'Units','normalized','Position',positions{i},'Enable','off');
end
uicontrol(fig,'Style','text', ...
    'String','[UI 미리보기] BLE 미연결 | L:데이터 대기(0) R:데이터 대기(0) | 전송대기:0, 확인:0', ...
    'Units','normalized','Position',[0.36 0.895 0.63 0.05], ...
    'BackgroundColor','w','HorizontalAlignment','left', ...
    'ForegroundColor',[0.15 0.15 0.15]);
panel = uipanel(fig,'BorderType','none','BackgroundColor','w', ...
    'Units','normalized','Position',[0 0.015 1 0.865]);
layout = tiledlayout(panel,2,3,'TileSpacing','compact','Padding','compact');
colors = lines(6);
for side = 1:2
    for col = 1:3
        ax = nexttile(layout,(side-1)*3+col);
        hold(ax,'on'); grid(ax,'on');
        set(ax,'Color','w'); xlabel(ax,'Time (s)');
        xlim(ax,[0 20]);
        if col == 1
            title(ax,sprintf('%s FSR (A0~A5)',handNames{side}));
            ylabel(ax,'ADC raw'); ylim(ax,[0 4095]);
            labels = arrayfun(@(k) sprintf('FSR%d',k),1:6,'UniformOutput',false);
        elseif col == 2
            title(ax,sprintf('%s IMU 가속도',handNames{side}));
            ylabel(ax,'g'); labels = {'AX','AY','AZ'};
        else
            title(ax,sprintf('%s IMU 자이로',handNames{side}));
            ylabel(ax,'deg/s'); labels = {'GX','GY','GZ'};
        end
        plotHandles = gobjects(1,numel(labels));
        for c = 1:numel(labels)
            plotHandles(c) = plot(ax,nan,nan,'Color',colors(c,:), 'LineWidth',1.2);
        end
        legend(ax,plotHandles,labels,'Location','northeast', ...
            'AutoUpdate','off','FontSize',7);
    end
end
end
