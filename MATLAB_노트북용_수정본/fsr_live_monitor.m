function fsr_live_monitor(port)
% FSR_LIVE_MONITOR  FSR/IMU 표시와 20회 동작 수집 및 HTTP 전송.
% 실행: fsr_live_monitor
% 포트 직접 지정: fsr_live_monitor("/dev/cu.usbserial-1440")
% Arduino IDE 시리얼 모니터를 먼저 닫으세요. 그래프 창을 닫으면 종료합니다.
% FSR 세로축 기본 범위: 0~500. IMU는 가속도(g)와 각속도(deg/s)를 나눕니다.

if nargin < 1 || strlength(string(port)) == 0
    ports = serialportlist("available");
    usbPorts = ports(contains(lower(ports), "usbserial") | ...
                     contains(lower(ports), "usbmodem"));
    if ismac
        cuPorts = usbPorts(startsWith(usbPorts, "/dev/cu."));
        ttyAliases = replace(cuPorts, "/dev/cu.", "/dev/tty.");
        usbPorts = usbPorts(~ismember(usbPorts, ttyAliases));
    end
    if numel(usbPorts) ~= 1
        error("FSR:SelectPort", ...
            ['USB 포트를 하나로 선택할 수 없습니다. 사용 가능한 포트: %s\n' ...
             'fsr_live_monitor("포트명")으로 직접 지정하세요.'], ...
            strjoin(ports, ", "));
    end
    port = usbPorts(1);
end
port = string(port);

pins = [25 26 34 33 32 35];
windowSeconds = 20;
fig = figure('Name', 'ESP32 FSR + IMU Live Monitor', 'NumberTitle', 'off', ...
    'Color', 'w', 'Position', [100 100 1100 750], 'Tag', 'FSR6LiveMonitor');
figureCleanup = onCleanup(@() closeFigure(fig));

s = serialport(port, 115200, 'Timeout', 1);
serialCleanup = onCleanup(@() delete(s));
flush(s, "input");

plotPanel = uipanel(fig, 'Units', 'normalized', 'Position', [0 0 1 0.82], ...
    'BorderType', 'none', 'BackgroundColor', 'w');
layout = tiledlayout(plotPanel, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
heading = title(layout, sprintf('FSR + IMU | %s | 115200 baud', port), ...
    'Interpreter', 'none', 'Color', [0.15 0.15 0.15]);
plotAxes = nexttile(layout, 1, [1 2]);
set(plotAxes, 'Color', 'w', 'XColor', [0.15 0.15 0.15], ...
    'YColor', [0.15 0.15 0.15]);
hold(plotAxes, 'on');
colors = lines(6);
curves = gobjects(1, 6);
legendLabels = cell(1, 6);
for k = 1:6
    curves(k) = animatedline(plotAxes, 'Color', colors(k, :), ...
        'LineWidth', 1.6, 'MaximumNumPoints', 2000);
    legendLabels{k} = sprintf('GPIO%d | ADC: --', pins(k));
end
xlabel(plotAxes, 'Time (s)');
ylabel(plotAxes, 'ADC raw');
title(plotAxes, 'FSR sensors', 'Color', [0.15 0.15 0.15]);
grid(plotAxes, 'on');
xlim(plotAxes, [0 windowSeconds]);
channelLegend = legend(plotAxes, curves, legendLabels, 'Location', 'northeast', ...
    'Interpreter', 'none', 'AutoUpdate', 'off', 'Color', 'w', ...
    'TextColor', [0.15 0.15 0.15]);
displayLimits = [50 100 200 500 1000 4095];
setYMaximum(plotAxes, 500);
uicontrol(fig, 'Style', 'text', 'String', 'Y max (ADC)', ...
    'Units', 'normalized', 'Position', [0.68 0.95 0.15 0.03], ...
    'BackgroundColor', 'w', 'ForegroundColor', [0.15 0.15 0.15], ...
    'HorizontalAlignment', 'right');
uicontrol(fig, 'Style', 'popupmenu', 'String', {'50','100','200','500','1000','4095'}, ...
    'Value', 4, 'Units', 'normalized', 'Position', [0.85 0.944 0.12 0.045], ...
    'BackgroundColor', 'w', 'ForegroundColor', [0.15 0.15 0.15], ...
    'Callback', @(source, ~) setYMaximum(plotAxes, displayLimits(source.Value)));

accelAxes = nexttile(layout, 3);
gyroAxes = nexttile(layout, 4);
imuAxes = [accelAxes gyroAxes];
for k = 1:2
    set(imuAxes(k), 'Color', 'w', 'XColor', [0.15 0.15 0.15], ...
        'YColor', [0.15 0.15 0.15]);
    hold(imuAxes(k), 'on');
    xlabel(imuAxes(k), 'Time (s)');
    grid(imuAxes(k), 'on');
    xlim(imuAxes(k), [0 windowSeconds]);
end
ylabel(accelAxes, 'Acceleration (g)');
ylabel(gyroAxes, 'Angular velocity (deg/s)');
accelHeading = title(accelAxes, 'IMU acceleration | Chip temp: -- C', ...
    'Color', [0.15 0.15 0.15]);
title(gyroAxes, 'IMU angular velocity', 'Color', [0.15 0.15 0.15]);
axisNames = {'X', 'Y', 'Z'};
accelCurves = gobjects(1, 3);
gyroCurves = gobjects(1, 3);
accelLabels = cell(1, 3);
gyroLabels = cell(1, 3);
for k = 1:3
    accelCurves(k) = animatedline(accelAxes, 'Color', colors(k, :), ...
        'LineWidth', 1.6, 'MaximumNumPoints', 2000);
    gyroCurves(k) = animatedline(gyroAxes, 'Color', colors(k, :), ...
        'LineWidth', 1.6, 'MaximumNumPoints', 2000);
    accelLabels{k} = sprintf('A%s | -- g', axisNames{k});
    gyroLabels{k} = sprintf('G%s | -- deg/s', axisNames{k});
end
accelLegend = legend(accelAxes, accelCurves, accelLabels, 'Location', 'northeast', ...
    'Interpreter', 'none', 'AutoUpdate', 'off', 'Color', 'w', ...
    'TextColor', [0.15 0.15 0.15]);
gyroLegend = legend(gyroAxes, gyroCurves, gyroLabels, 'Location', 'northeast', ...
    'Interpreter', 'none', 'AutoUpdate', 'off', 'Color', 'w', ...
    'TextColor', [0.15 0.15 0.15]);
allAxes = [plotAxes imuAxes];
allCurves = [curves accelCurves gyroCurves];
collector = fsr_http_collection(fig);
collectorCleanup = onCleanup(collector.close);

buffer = '';
synchronized = false;
firstMs = NaN;
previousMs = NaN;
samples = 0;
lastData = [];
screenUpdate = tic;
lastPlottedMs = NaN;  % Reduce GUI redraw work; collect every raw sensor frame.

while isgraphics(fig)
    available = s.NumBytesAvailable;
    if available > 0
        bytes = read(s, available, "uint8");
        buffer = [buffer reshape(char(bytes), 1, [])]; %#ok<AGROW>
        endings = find(buffer == newline);
        startIndex = 1;
        for stopIndex = endings
            line = strtrim(buffer(startIndex:stopIndex - 1));
            startIndex = stopIndex + 1;
            if ~synchronized
                synchronized = true; % 첫 줄은 중간부터 수신했을 수 있어 버립니다.
                continue;
            end
            fields = split(string(line), ",");
            if numel(fields) ~= 14
                continue;
            end
            values = str2double(fields(1:7)); % ms + FSR 6개
            if any(isnan(values) | isinf(values)) || any(values ~= fix(values)) || ...
                    values(1) < 0 || any(values(2:7) < 0 | values(2:7) > 4095)
                continue;
            end
            imuValues = str2double(fields(8:14)); % AX/AY/AZ, GX/GY/GZ, 온도
            imuValues(isinf(imuValues)) = NaN; % 결측값을 0으로 대체하지 않습니다.

            milliseconds = values(1);
            if isnan(firstMs) || milliseconds < previousMs
                firstMs = milliseconds;
                lastPlottedMs = NaN;
                for k = 1:numel(allCurves)
                    clearpoints(allCurves(k));
                end
            end
            previousMs = milliseconds;
            elapsed = (milliseconds - firstMs) / 1000;
            % Always save full-rate raw samples, but redraw charts ~4 times/s.
            if isnan(lastPlottedMs) || milliseconds - lastPlottedMs >= 250
                for k = 1:6
                    addpoints(curves(k), elapsed, values(k + 1));
                    legendLabels{k} = sprintf('GPIO%d | ADC: %d', pins(k), values(k + 1));
                end
                channelLegend.String = legendLabels;
                for k = 1:3
                    addpoints(accelCurves(k), elapsed, imuValues(k));
                    addpoints(gyroCurves(k), elapsed, imuValues(k + 3));
                    accelLabels{k} = sprintf('A%s | %.3f g', axisNames{k}, imuValues(k));
                    gyroLabels{k} = sprintf('G%s | %.2f deg/s', axisNames{k}, imuValues(k + 3));
                end
                accelLegend.String = accelLabels;
                gyroLegend.String = gyroLabels;
                accelHeading.String = sprintf('IMU acceleration | Chip temp: %.1f C', imuValues(7));
                set(allAxes, 'XLim', [max(0, elapsed - windowSeconds), ...
                                     max(windowSeconds, elapsed)]);
                lastPlottedMs = milliseconds;
            end
            collector.addSample(milliseconds, values(2:7), imuValues(1:6));
            samples = samples + 1;
            lastData = tic;
        end
        buffer = buffer(startIndex:end);
        if numel(buffer) > 8192
            buffer = ''; % 줄바꿈 없는 깨진 스트림의 메모리 증가를 제한합니다.
        end
    end

    if toc(screenUpdate) >= 0.25
        collector.tick();
        if isempty(lastData)
            state = 'Waiting for data';
        elseif toc(lastData) > 2
            state = sprintf('No new data for %.1f s', toc(lastData));
        else
            state = sprintf('Receiving | Samples: %d', samples);
        end
        heading.String = sprintf('FSR + IMU | %s | 115200 baud | %s', port, state);
        drawnow limitrate;
        screenUpdate = tic;
    end
    pause(0.02);
end
end

function closeFigure(fig)
if isgraphics(fig)
    delete(fig);
end
end

function setYMaximum(axesHandles, maximum)
for k = 1:numel(axesHandles)
    ylim(axesHandles(k), [0 maximum]);
    if maximum == 4095
        yticks(axesHandles(k), [0 1024 2048 3072 4095]);
    else
        yticks(axesHandles(k), linspace(0, maximum, 6));
    end
end
end
