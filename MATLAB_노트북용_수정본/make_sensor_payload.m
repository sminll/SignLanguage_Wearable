function payload = make_sensor_payload(fsr, imu, deviceId, sequence, timeMs, source, subjectId, sessionId)
% Build one raw sensor record with optional subject/session identifiers.
if nargin < 7, subjectId = ''; end
if nargin < 8, sessionId = ''; end
if isstring(subjectId), subjectId = char(subjectId); end
if isstring(sessionId), sessionId = char(sessionId); end
if ~ischar(subjectId) || ~ischar(sessionId)
    error('sensor:InvalidMetadata', 'Subject and session must be text.');
end
if ~isnumeric(fsr) || ~isreal(fsr) || ~isvector(fsr) || numel(fsr) ~= 6 ...
        || any(isnan(fsr(:)) | isinf(fsr(:))) || any(fsr(:) < 0)
    error('sensor:InvalidFSR', 'FSR must contain six finite nonnegative raw ADC values.');
end
if ~isnumeric(imu) || ~isreal(imu) || ~isvector(imu) || numel(imu) ~= 6
    error('sensor:InvalidIMU', 'IMU must contain six real numeric values.');
end
if isstring(deviceId) && isscalar(deviceId)
    deviceId = char(deviceId);
end
if ~ischar(deviceId) || ~isscalar(deviceId) ...
        || ~(deviceId == 'L' || deviceId == 'R')
    error('sensor:InvalidDevice', 'Device ID must be L or R.');
end
if ~isnumeric(sequence) || ~isreal(sequence) || ~isscalar(sequence) ...
        || isnan(sequence) || isinf(sequence) || sequence < 1 ...
        || sequence ~= fix(sequence)
    error('sensor:InvalidSequence', 'Sequence must be a positive integer.');
end
if ~isnumeric(timeMs) || ~isreal(timeMs) || ~isscalar(timeMs) ...
        || isnan(timeMs) || isinf(timeMs) || timeMs < 0 || timeMs ~= fix(timeMs)
    error('sensor:InvalidTime', 'Time must be nonnegative integer milliseconds since boot.');
end
if isstring(source) && isscalar(source)
    source = char(source);
end
if ~ischar(source) || (~isrow(source) && ~isempty(source))
    error('sensor:InvalidSource', 'Source must be a character row or scalar string.');
end

fsr = double(fsr(:));
imu = double(imu(:));
imuOk = double(all(~isnan(imu) & ~isinf(imu)));
imu(isnan(imu) | isinf(imu)) = NaN;
payload = struct('source', source, 'device_id', deviceId, ...
    'seq', double(sequence), 'time_ms', double(timeMs), ...
    'fsr1_raw', fsr(1), 'fsr2_raw', fsr(2), 'fsr3_raw', fsr(3), ...
    'fsr4_raw', fsr(4), 'fsr5_raw', fsr(5), 'fsr6_raw', fsr(6), ...
    'ax_g', imu(1), 'ay_g', imu(2), 'az_g', imu(3), ...
    'gx_dps', imu(4), 'gy_dps', imu(5), 'gz_dps', imu(6), ...
    'imu_ok', imuOk, 'subject_id', subjectId, 'session_id', sessionId);
end
