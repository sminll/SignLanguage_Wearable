function phase = sensor_collection_phase(elapsedMs)
% Twenty fixed trials: collect for five seconds, then rest for three.
if ~isnumeric(elapsedMs) || ~isreal(elapsedMs) || ~isscalar(elapsedMs) ...
        || isnan(elapsedMs) || isinf(elapsedMs) || elapsedMs < 0
    error('sensor:InvalidElapsedTime', 'Elapsed time must be finite nonnegative milliseconds.');
end
elapsedMs = double(elapsedMs);
if elapsedMs >= 160000
    phase = struct('trial', 20, 'collecting', false, ...
        'remainingSec', 0, 'done', true);
    return;
end
trial = floor(elapsedMs / 8000) + 1;
withinTrial = mod(elapsedMs, 8000);
collecting = withinTrial < 5000;
if collecting
    remainingSec = (5000 - withinTrial) / 1000;
else
    remainingSec = (8000 - withinTrial) / 1000;
end
phase = struct('trial', trial, 'collecting', collecting, ...
    'remainingSec', remainingSec, 'done', false);
end
