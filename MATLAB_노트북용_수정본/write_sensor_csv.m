function write_sensor_csv(filePath, samples, mode)
% Persist flat sensor records; replace writes a complete snapshot first.
if isstring(filePath) && isscalar(filePath)
    filePath = char(filePath);
end
if ~ischar(filePath) || ~isrow(filePath) || isempty(filePath)
    error('sensor:InvalidCsvPath', 'CSV path must be a nonempty character row or scalar string.');
end
if isfolder(filePath)
    error('sensor:InvalidCsvPath', 'CSV path must name a file, not a directory.');
end
if isstring(mode) && isscalar(mode)
    mode = char(mode);
end
if ~ischar(mode) || ~isrow(mode) || ~any(strcmp(mode, {'append', 'replace'}))
    error('sensor:InvalidCsvMode', 'CSV mode must be append or replace.');
end
if ~iscell(samples) || (~isvector(samples) && ~isempty(samples))
    error('sensor:InvalidCsvSamples', 'Samples must be a cell vector of sensor payloads.');
end
fields = {'source', 'device_id', 'seq', 'time_ms', ...
    'fsr1_raw', 'fsr2_raw', 'fsr3_raw', 'fsr4_raw', 'fsr5_raw', ...
    'fsr6_raw', 'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', ...
    'gz_dps', 'imu_ok', 'subject_id', 'session_id'};
for k = 1:numel(samples)
    if ~isstruct(samples{k}) || ~isscalar(samples{k}) ...
            || ~all(isfield(samples{k}, fields))
        error('sensor:InvalidCsvSamples', 'Each sample must be a scalar sensor payload.');
    end
end

fid = -1;
temporaryPath = '';
if strcmp(mode, 'replace')
    folder = fileparts(filePath);
    if isempty(folder)
        folder = pwd;
    end
    temporaryPath = [tempname(folder) '.csv'];
    outputPath = temporaryPath;
    permission = 'w';
    needsHeader = true;
else
    outputPath = filePath;
    permission = 'a';
    info = dir(filePath);
    needsHeader = isempty(info) || info.bytes == 0;
end
[fid, message] = fopen(outputPath, permission, 'n', 'UTF-8');
if fid < 0
    error('sensor:CsvOpenFailed', 'Cannot open CSV file: %s', message);
end
try
if needsHeader
    fprintf(fid, '%s\n', strjoin(fields, ','));
end
for k = 1:numel(samples)
    sample = samples{k};
    numbers = zeros(1, 15);
    for field = 3:17
        value = sample.(fields{field});
        if ~isnumeric(value) || ~isreal(value) || ~isscalar(value)
            error('sensor:InvalidCsvSamples', 'Sensor values must be real numeric scalars.');
        end
        numbers(field - 2) = value;
    end
    fprintf(fid, '%s,%s', quoteField(sample.source), quoteField(sample.device_id));
    fprintf(fid, ',%.17g', numbers);
    fprintf(fid, ',%s,%s\n', quoteField(sample.subject_id), quoteField(sample.session_id));
end
[message, errorNumber] = ferror(fid);
if errorNumber ~= 0
    error('sensor:CsvWriteFailed', 'Cannot write CSV file: %s', message);
end
closeStatus = fclose(fid);
fid = -1;
if closeStatus ~= 0
    error('sensor:CsvWriteFailed', 'Cannot finish writing CSV file.');
end
if strcmp(mode, 'replace')
    [success, message] = movefile(temporaryPath, filePath, 'f');
    if ~success
        error('sensor:CsvReplaceFailed', 'Cannot replace CSV file: %s', message);
    end
end

catch exception
    if fid >= 0
        try, fclose(fid); catch, end
    end
    if ~isempty(temporaryPath) && isfile(temporaryPath)
        delete(temporaryPath);
    end
    rethrow(exception);
end
end

function quoted = quoteField(value)
if isstring(value) && isscalar(value)
    value = char(value);
end
if ~ischar(value) || (~isrow(value) && ~isempty(value))
    error('sensor:InvalidCsvSamples', 'CSV text fields must be character rows or scalar strings.');
end
quoted = ['"' strrep(value, '"', '""') '"'];
end
