function result = post_sensor_batch(endpoint, batch, retryServerError)
% Retry one HTTP 500 only when the caller knows the receiver deduplicates
% (session_id, hand, seq). Never retry a timeout or conflict automatically.
if nargin < 3, retryServerError = false; end
if ~iscell(batch) || (~isvector(batch) && ~isempty(batch)) ...
        || any(~cellfun(@(payload) isstruct(payload) && isscalar(payload), batch(:)))
    error('sensor:InvalidBatch', 'Batch must be a cell vector of scalar payload structs.');
end
result = struct('sent', 0, 'unconfirmed', numel(batch), 'message', '');
if isempty(batch)
    return;
end
options = weboptions('MediaType', 'application/json', ...
    'ContentType', 'text', 'Timeout', 10);
checkedV3=false;
for k = 1:numel(batch)
    try
        sample = batch{k};
        % Convert MATLAB's 17-column sensor format into the receiver schema.
        if strcmp(sample.device_id, 'L')
            hand = 'left';
        elseif strcmp(sample.device_id, 'R')
            hand = 'right';
        else
            error('sensor:InvalidDevice', 'Unknown device_id: %s', sample.device_id);
        end
        label = regexprep(sample.source, '[0-9]+$', '');
        payload = struct( ...
            'source', 'REAL', ...
            'subject_id', sample.subject_id, ...
            'session_id', sample.session_id, ...
            'seq', sample.seq, ...
            'hand', hand, ...
            'trial_id', sample.source, ...
            'label', label, ...
            'time_ms', sample.time_ms, ...
            'fsr1', sample.fsr1_raw, 'fsr2', sample.fsr2_raw, ...
            'fsr3', sample.fsr3_raw, 'fsr4', sample.fsr4_raw, ...
            'fsr5', sample.fsr5_raw, 'fsr6', sample.fsr6_raw, ...
            'ax', sample.ax_g, 'ay', sample.ay_g, 'az', sample.az_g, ...
            'gx', sample.gx_dps, 'gy', sample.gy_dps, 'gz', sample.gz_dps);
        extra = {'gesture_type','host_elapsed_ms','trial_elapsed_ms'};
        if any(isfield(sample,extra))
            if ~all(isfield(sample,extra))
                error('sensor:TimingMetadata','양손 시간 메타데이터가 누락됐습니다.');
            end
            if ~checkedV3
                healthUrl=regexprep(endpoint,'/sensor$','/health');
                health=webread(healthUrl,weboptions('Timeout',5));
                if ~isstruct(health) || ~isfield(health,'schema_versions') || ~ismember(3,health.schema_versions)
                    error('sensor:ServerVersion','서버를 v3 수신 지원 버전으로 업데이트하세요. CSV는 보존됩니다.');
                end
                checkedV3=true;
            end
            for field=extra, payload.(field{1})=sample.(field{1}); end
        end
        try
            webwrite(endpoint, payload, options);
        catch postError
            if retryServerError && ~isempty(sample.session_id) ...
                    && contains(postError.identifier, 'HTTP500')
                % Keep the entire payload and idempotency key identical.
                webwrite(endpoint, payload, options);
            else
                rethrow(postError);
            end
        end
        result.sent = result.sent + 1;
        result.unconfirmed = numel(batch) - result.sent;
    catch exception
        % A server may have accepted a request whose response was lost.
        % Leave it unconfirmed and do not retry, to avoid duplicate records.
        result.message = exception.message;
        return;
    end
end
end
