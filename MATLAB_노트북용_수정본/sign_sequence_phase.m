function phase = sign_sequence_phase(elapsedMs, attempts)
% Two explicit windows per attempt: A 5s / rest 3s / B 5s / rest 3s.
validateattributes(elapsedMs,{'numeric'},{'scalar','finite','real','nonnegative'});
validateattributes(attempts,{'numeric'},{'scalar','integer','positive'});
phase.done = elapsedMs >= attempts*16000;
phase.attempt = min(attempts,floor(elapsedMs/16000)+1);
phase.step = min(2,floor(mod(elapsedMs,16000)/8000)+1);
phase.windowMs = mod(elapsedMs,8000);
phase.collecting = ~phase.done && phase.windowMs < 5000;
phase.windowStartMs = (phase.attempt-1)*16000+(phase.step-1)*8000;
end
