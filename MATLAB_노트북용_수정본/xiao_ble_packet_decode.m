function [state, sample] = xiao_ble_packet_decode(state, packet)
% XIAO_BLE_PACKET_DECODE  Assemble one XIAO 3-fragment BLE sensor sample.
% Matches sensor_packet.h exactly (20 bytes x 3, 45-byte body).
% state = []; initially. sample = [] until all three fragments arrive.
%
% Frame layout: [marker=0xA0..0xA2] [sequence uint32 LE] [15 body bytes].
% Body: time_ms uint32 LE, 6 FSR uint16 LE, 7 IMU float32 LE,
%       1 imu_ok byte. Missing IMU fields are IEEE-754 NaN.

sample = [];
packet = uint8(packet(:).');
if numel(packet) ~= 20 || ~ismember(packet(1), uint8([160 161 162]))
    return;
end
piece = double(packet(1)) - 159; % 1/2/3 for A0/A1/A2
seq = readU32LE(packet(2:5));

if isempty(state)
    state = freshFrame(seq);
elseif seq ~= state.seq
    % Only accept a newer frame, accounting for uint32 rollover.
    forward = mod(seq - state.seq, 2^32);
    if forward >= 2^31
        return; % A delayed fragment from an older frame.
    end
    state = freshFrame(seq); % Discard incomplete prior frame.
end

if state.have(piece)
    return; % Repeated fragment.
end
first = (piece - 1) * 15 + 1;
state.body(first:first + 14) = packet(6:20);
state.have(piece) = true;
if ~all(state.have)
    return;
end

body = state.body;
fsr = zeros(1,6);
for k = 1:6
    p = 5 + (k-1)*2;
    fsr(k) = double(body(p)) + 256*double(body(p+1));
end
imu = zeros(1,7);
for k = 1:7
    p = 17 + (k-1)*4;
    bits = uint32(readU32LE(body(p:p+3)));
    imu(k) = double(typecast(bits, 'single'));
end
sample = struct('sequence', seq, 'time_ms', readU32LE(body(1:4)), ...
    'fsr', fsr, 'imu', imu, 'imu_ok', body(45) == 1);
if ~sample.imu_ok
    sample.imu(:) = NaN;
end
state = []; % Complete frame: wait for the next sequence number.
end

function state = freshFrame(seq)
state = struct('seq', seq, 'have', false(1,3), 'body', zeros(1,45,'uint8'));
end

function value = readU32LE(bytes)
bytes = double(bytes);
value = bytes(1) + 256*bytes(2) + 65536*bytes(3) + 16777216*bytes(4);
end
