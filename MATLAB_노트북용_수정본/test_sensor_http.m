function tests = test_sensor_http
% Pure payload and collection-schedule checks. No serial ports or HTTP calls.
tests = functiontests(localfunctions);
end

function testPayloadFieldOrder(testCase)
payload = make_sensor_payload(1:6, 11:16, 'L', 1, 0, '가위1');
expected = {'source'; 'device_id'; 'seq'; 'time_ms'; ...
    'fsr1_raw'; 'fsr2_raw'; 'fsr3_raw'; 'fsr4_raw'; 'fsr5_raw'; ...
    'fsr6_raw'; 'ax_g'; 'ay_g'; 'az_g'; 'gx_dps'; 'gy_dps'; ...
    'gz_dps'; 'imu_ok'; 'subject_id'; 'session_id'};
verifyTrue(testCase, isscalar(payload));
verifyEqual(testCase, fieldnames(payload), expected);
end

function testPayloadValuesAndKoreanSourceRoundtrip(testCase)
payload = make_sensor_payload([11 22 33 44 55 66], ...
    [0.25 -0.5 1 2 -3 4], 'R', 7, 1234, '가위1');
expected = struct('source', '가위1', 'device_id', 'R', ...
    'seq', 7, 'time_ms', 1234, 'fsr1_raw', 11, 'fsr2_raw', 22, ...
    'fsr3_raw', 33, 'fsr4_raw', 44, 'fsr5_raw', 55, 'fsr6_raw', 66, ...
    'ax_g', 0.25, 'ay_g', -0.5, 'az_g', 1, 'gx_dps', 2, ...
    'gy_dps', -3, 'gz_dps', 4, 'imu_ok', 1, ...
    'subject_id', '', 'session_id', '');
verifyEqual(testCase, payload, expected);
verifyClass(testCase, payload.source, 'char');
verifyClass(testCase, payload.imu_ok, 'double');
verifyEqual(testCase, jsondecode(jsonencode(payload)), expected);
end

function testExplicitSubjectAndSession(testCase)
payload = make_sensor_payload(1:6, zeros(1,6), 'R', 10, 3000, ...
    '아프다2', 'subject3', 's20261031_123000_123456789');
verifyEqual(testCase, payload.subject_id, 'subject3');
verifyEqual(testCase, payload.session_id, 's20261031_123000_123456789');
end

function testNonfiniteImuBecomesJsonNull(testCase)
payload = make_sensor_payload(1:6, [NaN Inf -Inf 0.25 0 -3], ...
    'L', 1, 0, '시험');
verifyTrue(testCase, all(isnan([payload.ax_g payload.ay_g payload.az_g])));
verifyEqual(testCase, [payload.gx_dps payload.gy_dps payload.gz_dps], ...
    [0.25 0 -3]);
verifyEqual(testCase, payload.imu_ok, 0);
encoded = jsonencode(payload);
verifyTrue(testCase, contains(encoded, '"ax_g":null'));
verifyTrue(testCase, contains(encoded, '"ay_g":null'));
verifyTrue(testCase, contains(encoded, '"az_g":null'));
verifyFalse(testCase, contains(encoded, 'NaN'));
verifyFalse(testCase, contains(encoded, 'Inf'));
end

function testEveryImuAxisControlsImuOk(testCase)
for axis = 1:6
    imu = zeros(1, 6);
    imu(axis) = NaN;
    payload = make_sensor_payload(1:6, imu, 'L', 1, 0, '시험');
    verifyEqual(testCase, payload.imu_ok, 0);
end
end

function testPayloadRejectsInvalidInputs(testCase)
verifyError(testCase, @() make_sensor_payload(1:5, zeros(1, 6), ...
    'L', 1, 0, '시험'), 'sensor:InvalidFSR');
verifyError(testCase, @() make_sensor_payload([1 2 3 4 5 NaN], ...
    zeros(1, 6), 'L', 1, 0, '시험'), 'sensor:InvalidFSR');
verifyError(testCase, @() make_sensor_payload([-1 2 3 4 5 6], ...
    zeros(1, 6), 'L', 1, 0, '시험'), 'sensor:InvalidFSR');
verifyError(testCase, @() make_sensor_payload(1:6, zeros(1, 6), ...
    'X', 1, 0, '시험'), 'sensor:InvalidDevice');
verifyError(testCase, @() make_sensor_payload(1:6, zeros(1, 6), ...
    'L', 0, 0, '시험'), 'sensor:InvalidSequence');
verifyError(testCase, @() make_sensor_payload(1:6, zeros(1, 6), ...
    'L', 1.5, 0, '시험'), 'sensor:InvalidSequence');
verifyError(testCase, @() make_sensor_payload(1:6, zeros(1, 6), ...
    'L', 1, -1, '시험'), 'sensor:InvalidTime');
verifyError(testCase, @() make_sensor_payload(1:6, zeros(1, 6), ...
    'L', 1, Inf, '시험'), 'sensor:InvalidTime');
end

function testCollectionPhaseBoundaries(testCase)
elapsed = [0 4999 5000 7999 8000 159999 160000];
trials = [1 1 1 1 2 20 20];
collecting = [true true false false true false false];
remaining = [5 0.001 3 0.001 5 0.001 0];
done = [false false false false false false true];
for k = 1:numel(elapsed)
    phase = sensor_collection_phase(elapsed(k));
    verifyEqual(testCase, phase.trial, trials(k));
    verifyEqual(testCase, phase.collecting, collecting(k));
    verifyClass(testCase, phase.collecting, 'logical');
    verifyEqual(testCase, phase.remainingSec, remaining(k), 'AbsTol', 1e-12);
    verifyGreaterThanOrEqual(testCase, phase.remainingSec, 0);
    verifyEqual(testCase, phase.done, done(k));
    verifyClass(testCase, phase.done, 'logical');
end
end

function testCollectionPhaseStaysDone(testCase)
phase = sensor_collection_phase(200000);
verifyEqual(testCase, phase, struct('trial', 20, 'collecting', false, ...
    'remainingSec', 0, 'done', true));
end

function testCollectionPhaseRejectsInvalidElapsedTime(testCase)
verifyError(testCase, @() sensor_collection_phase(-1), ...
    'sensor:InvalidElapsedTime');
verifyError(testCase, @() sensor_collection_phase(NaN), ...
    'sensor:InvalidElapsedTime');
end

function testEmptyBatchDoesNotPost(testCase)
result = post_sensor_batch('', {});
verifyEqual(testCase, result, ...
    struct('sent', 0, 'unconfirmed', 0, 'message', ''));
end

function testInvalidBatchRejectedWithoutPosting(testCase)
verifyError(testCase, @() post_sensor_batch('', struct('value', 1)), ...
    'sensor:InvalidBatch');
verifyError(testCase, @() post_sensor_batch('', {struct('value', 1), 2}), ...
    'sensor:InvalidBatch');
end
