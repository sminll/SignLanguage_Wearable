function test_xiao_ble_packet_decode()
% Run without XIAO or BLE hardware to check the byte layout.
body = zeros(1,45,'uint8');
body(1:4) = uint8([210 4 0 0]); % 1234 ms
fsr = [0 1 4095 123 2000 42];
for k = 1:6
    n = uint16(fsr(k));
    body(5+(k-1)*2 : 6+(k-1)*2) = typecast(n, 'uint8');
end
imu = single([0.25 -0.5 1.0 45 -12 8 29.5]);
for k = 1:7
    p = 17 + (k-1)*4;
    body(p:p+3) = typecast(imu(k), 'uint8');
end
body(45) = 1;
seq = uint8([7 0 0 0]);
packets = zeros(3,20,'uint8');
for k = 1:3
    packets(k,:) = [uint8(159+k) seq body((k-1)*15+1:k*15)];
end
state = [];
for k = 1:2
    [state, sample] = xiao_ble_packet_decode(state, packets(k,:));
    assert(isempty(sample));
end
[state, sample] = xiao_ble_packet_decode(state, packets(3,:));
assert(sample.sequence == 7 && sample.time_ms == 1234);
assert(isequal(sample.fsr, fsr));
assert(all(abs(sample.imu-double(imu)) < 1e-6));
assert(sample.imu_ok);
assert(isempty(state));
% Missing the middle packet must not make a false sample.
[state, sample] = xiao_ble_packet_decode([], packets(1,:));
assert(isempty(sample));
[~, sample] = xiao_ble_packet_decode(state, packets(3,:));
assert(isempty(sample));
fprintf('PASS: XIAO BLE 3-packet decoder layout and incomplete-frame test.\n');
end
