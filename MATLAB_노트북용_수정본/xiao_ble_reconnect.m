% XIAO_BLE_RECONNECT Reconnect the single-board monitor after a power change.
% Set the board address from blelist before running this script:
%   xiao_ble_address = "YOUR_BLE_ADDRESS";
%   xiao_ble_reconnect
% Replaces workspace variables b/c, and closes only single-board test figures.
% Opens the monitor; capture and upload require explicit button actions.

if ~exist('xiao_ble_address', 'var')
    error('XIAO:Address', ...
        'blelist로 주소를 확인한 뒤 xiao_ble_address = "장치주소"; 를 실행하세요.');
end
validateattributes(xiao_ble_address, {'char','string'}, {'nonempty'});
xiao_ble_address = string(xiao_ble_address);
if ~isscalar(xiao_ble_address) || ismissing(xiao_ble_address) || ...
        strlength(strtrim(xiao_ble_address)) == 0
    error('XIAO:Address', 'BLE 장치 주소 하나를 입력하세요.');
end

% Close the preview first so its nested callback releases the old characteristic.
close(findall(groot, 'Type', 'figure', ...
    'Name', 'XIAO BLE 양손 센서 모니터 (1개 보드 실시간 테스트)'));
if ~isempty(findall(groot,'Type','figure', ...
        'Name','XIAO BLE 양손 센서 모니터 (1개 보드 실시간 테스트)'))
    error('XIAO:MonitorOpen', ...
        '기존 모니터가 닫히지 않았습니다. 저장/전송 상태를 확인한 후 다시 연결하세요.');
end
clear c b

b = ble(xiao_ble_address);
c = characteristic(b, ...
    "19B10010-E8F2-537E-4F6C-D104768A1214", ...
    "19B10011-E8F2-537E-4F6C-D104768A1214");
subscribe(c, "notification");
xiao_ble_single_ui_test(c);
