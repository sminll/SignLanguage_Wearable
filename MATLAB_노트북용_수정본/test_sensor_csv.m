function tests = test_sensor_csv
% CSV checks use temporary files only. No serial ports or HTTP calls.
tests = functiontests(localfunctions);
end

function setup(testCase)
testCase.TestData.folder = tempname(tempdir);
mkdir(testCase.TestData.folder);
testCase.TestData.filePath = fullfile(testCase.TestData.folder, 'samples.csv');
end

function teardown(testCase)
if isfolder(testCase.TestData.folder)
    rmdir(testCase.TestData.folder, 's');
end
end

function testHeaderAndRawValues(testCase)
payload = make_sensor_payload([11 22 33 44 55 66], ...
    [0.125 -0.5000000000000001 1 2 -3 4], 'R', 7, 1234, '가위1');
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
lines = csvLines(testCase);
expected = ['source,device_id,seq,time_ms,fsr1_raw,fsr2_raw,' ...
    'fsr3_raw,fsr4_raw,fsr5_raw,fsr6_raw,ax_g,ay_g,az_g,' ...
    'gx_dps,gy_dps,gz_dps,imu_ok,subject_id,session_id'];
verifyEqual(testCase, lines{1}, expected);
verifyEqual(testCase, numel(strsplit(lines{1}, ',')), 19);
data = readCsv(testCase);
verifyEqual(testCase, width(data), 19);
verifyEqual(testCase, height(data), 1);
verifyEqual(testCase, data{1, 3:17}, ...
    [7 1234 11 22 33 44 55 66 0.125 -0.5000000000000001 1 2 -3 4 1]);
verifyEqual(testCase, data.device_id, "R");
end

function testSessionMetadataSavedInRawCsv(testCase)
payload = make_sensor_payload(1:6, zeros(1, 6), 'R', 5, 12345, ...
    '못하다1', 'subject3', 's20261031_123000_123456789');
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
data = readCsv(testCase);
verifyEqual(testCase, data.subject_id, "subject3");
verifyEqual(testCase, data.session_id, "s20261031_123000_123456789");
end

function testAppendKeepsOneHeaderAndEveryRow(testCase)
first = make_sensor_payload(1:6, zeros(1, 6), 'L', 1, 0, '가위1');
second = make_sensor_payload(7:12, ones(1, 6), 'L', 2, 100, '가위1');
third = make_sensor_payload(13:18, 2 * ones(1, 6), 'L', 3, 200, '가위2');
write_sensor_csv(testCase.TestData.filePath, {first}, 'append');
write_sensor_csv(testCase.TestData.filePath, {second; third}, 'append');
lines = csvLines(testCase);
verifyEqual(testCase, numel(lines), 4);
verifyEqual(testCase, sum(strcmp(lines, lines{1})), 1);
data = readCsv(testCase);
verifyEqual(testCase, data.seq, [1; 2; 3]);
verifyEqual(testCase, data.fsr1_raw, [1; 7; 13]);
end

function testKoreanCommaAndQuotesRoundtrip(testCase)
source = '가위, "오른손"1';
payload = make_sensor_payload(1:6, zeros(1, 6), 'R', 1, 0, source);
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
lines = csvLines(testCase);
verifyTrue(testCase, startsWith(lines{2}, '"가위, ""오른손""1","R",'));
data = readCsv(testCase);
verifyEqual(testCase, data.source, string(source));
end

function testMissingImuStaysNaN(testCase)
payload = make_sensor_payload(1:6, [NaN Inf -Inf 0.25 0 -3], ...
    'L', 1, 0, '시험');
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
lines = csvLines(testCase);
fields = strsplit(lines{2}, ',');
verifyEqual(testCase, fields(11:13), {'NaN', 'NaN', 'NaN'});
data = readCsv(testCase);
verifyTrue(testCase, all(isnan(data{1, 11:13})));
verifyEqual(testCase, data{1, 14:17}, [0.25 0 -3 0]);
end

function testReplaceRemovesLatestTrialFromSnapshot(testCase)
first = make_sensor_payload(1:6, zeros(1, 6), 'L', 1, 0, '가위1');
second = make_sensor_payload(7:12, ones(1, 6), 'L', 2, 100, '가위1');
latest = make_sensor_payload(13:18, ones(1, 6), 'L', 3, 8000, '가위2');
write_sensor_csv(testCase.TestData.filePath, {first, second, latest}, 'append');
write_sensor_csv(testCase.TestData.filePath, {first, second}, 'replace');
data = readCsv(testCase);
verifyEqual(testCase, data.seq, [1; 2]);
verifyEqual(testCase, data.source, ["가위1"; "가위1"]);
verifyEqual(testCase, numel(csvLines(testCase)), 3);
verifyEqual(testCase, numel(dir(fullfile(testCase.TestData.folder, '*.csv'))), 1);
end

function testEmptyReplaceKeepsOnlyHeader(testCase)
payload = make_sensor_payload(1:6, zeros(1, 6), 'L', 1, 0, '가위1');
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
write_sensor_csv(testCase.TestData.filePath, {}, 'replace');
lines = csvLines(testCase);
verifyEqual(testCase, numel(lines), 1);
verifyEqual(testCase, numel(strsplit(lines{1}, ',')), 19);
data = readCsv(testCase);
verifyEqual(testCase, height(data), 0);
verifyEqual(testCase, width(data), 19);
end

function testOpenFailureIsReportedWithoutCreatingFolder(testCase)
filePath = fullfile(testCase.TestData.folder, 'missing', 'samples.csv');
payload = make_sensor_payload(1:6, zeros(1, 6), 'L', 1, 0, '시험');
verifyError(testCase, @() write_sensor_csv(filePath, {payload}, 'append'), ...
    'sensor:CsvOpenFailed');
verifyFalse(testCase, isfolder(fileparts(filePath)));
end

function testInvalidReplacePreservesExistingFile(testCase)
payload = make_sensor_payload(1:6, zeros(1, 6), 'L', 1, 0, '시험');
write_sensor_csv(testCase.TestData.filePath, {payload}, 'append');
before = csvLines(testCase);
invalid = payload;
invalid.ax_g = [1 2];
verifyError(testCase, @() write_sensor_csv(testCase.TestData.filePath, ...
    {payload, invalid}, 'replace'), 'sensor:InvalidCsvSamples');
verifyEqual(testCase, csvLines(testCase), before);
verifyEqual(testCase, numel(dir(fullfile(testCase.TestData.folder, '*.csv'))), 1);
end

function lines = csvLines(testCase)
fid = fopen(testCase.TestData.filePath, 'r');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid, Inf, '*uint8').';
text = native2unicode(bytes, 'UTF-8');
lines = regexp(strtrim(text), '\r?\n', 'split');
end

function data = readCsv(testCase)
data = readtable(testCase.TestData.filePath, 'FileType', 'text', ...
    'Delimiter', ',', 'TextType', 'string', 'Encoding', 'UTF-8');
end
