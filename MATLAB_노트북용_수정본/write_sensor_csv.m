function write_sensor_csv(filePath, samples, mode, schemaVersion)
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
extra = {'gesture_type','host_elapsed_ms','trial_elapsed_ms'};
timed = ~isempty(samples) && any(isfield(samples{1},extra));
if nargin>=4
    validateattributes(schemaVersion,{'numeric'},{'scalar','integer'});
    if ~ismember(schemaVersion,[2 3]), error('sensor:CsvSchema','CSV 버전은 2 또는 3입니다.'); end
    timed = schemaVersion==3;
end
if timed, fields = [fields extra]; end
for k = 1:numel(samples)
    if ~isstruct(samples{k}) || ~isscalar(samples{k}) ...
            || ~all(isfield(samples{k}, fields))
        error('sensor:InvalidCsvSamples', 'Each sample must be a scalar sensor payload.');
    end
    if any(isfield(samples{k},extra)) ~= timed
        error('sensor:InvalidCsvSamples','구형/양손 시간 형식을 섞어 저장할 수 없습니다.');
    end
    if timed
        validatestring(samples{k}.gesture_type,{'left_only','right_only','both'});
        validateattributes(samples{k}.host_elapsed_ms,{'numeric'},{'scalar','real','finite','nonnegative'});
        validateattributes(samples{k}.trial_elapsed_ms,{'numeric'},{'scalar','real','finite','nonnegative','<',5000});
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
    if ~needsHeader
        readId=fopen(filePath,'r','n','UTF-8');
        if readId<0, error('sensor:CsvOpenFailed','CSV 헤더 확인 실패'); end
        header=fgetl(readId); fclose(readId);
        if ~strcmp(header,strjoin(fields,','))
            error('sensor:CsvSchemaMismatch','기존 CSV와 열 형식이 다릅니다.');
        end
    end
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
    fprintf(fid, ',%s,%s', quoteField(sample.subject_id), quoteField(sample.session_id));
    if timed
        validateattributes(sample.host_elapsed_ms,{'numeric'},{'scalar','real','finite','nonnegative'});
        validateattributes(sample.trial_elapsed_ms,{'numeric'},{'scalar','real','finite','nonnegative','<',5000});
        fprintf(fid, ',%s,%.17g,%.17g',quoteField(sample.gesture_type), ...
            sample.host_elapsed_ms,sample.trial_elapsed_ms);
    end
    fprintf(fid,'\n');
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
