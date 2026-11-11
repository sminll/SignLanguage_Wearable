function xiao_ble_sequence_experiment(leftAddress,rightAddress,saveDirectory)
% Dedicated fixed-window sequence experiment using the existing BLE monitor.
if nargin<3, saveDirectory=fullfile(fileparts(mfilename('fullpath')),'sensor_data','sequences'); end
collector=sign_sequence_collection(saveDirectory);
cleanup=onCleanup(@() collector.close()); %#ok<NASGU>
xiao_ble_dual_monitor(leftAddress,rightAddress,collector.addFrame);
end
