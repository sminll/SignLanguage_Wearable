function tests = test_sensor_collection_review
% Invisible UI checks; no serial connection and never press Send.
tests = functiontests(localfunctions);
end

function setup(testCase)
testCase.TestData.saveDirectory = tempname;
mkdir(testCase.TestData.saveDirectory);
testCase.TestData.fig = figure('Visible', 'off');
testCase.TestData.collector = fsr_http_collection( ...
    testCase.TestData.fig, testCase.TestData.saveDirectory);
set(control(testCase, 'SensorGestureName'), 'String', '가위');
set(control(testCase, 'SensorHand'), 'Value', 2);
set(control(testCase, 'SensorSubject'), 'Value', 2);
press(testCase, 'SensorStart');
end

function teardown(testCase)
if isfield(testCase.TestData, 'collector')
    press(testCase, 'SensorStop');
    press(testCase, 'SensorClear');
    testCase.TestData.collector.close();
end
if isfield(testCase.TestData, 'fig') && isgraphics(testCase.TestData.fig)
    delete(testCase.TestData.fig);
end
if isfield(testCase.TestData, 'saveDirectory') && isfolder(testCase.TestData.saveDirectory)
    rmdir(testCase.TestData.saveDirectory, 's');
end
end

function testCaptureWaitsForManualSend(testCase)
sample(testCase, 1000);
sample(testCase, 2000);
verifyStatus(testCase, '보관 2');
verifyEqual(testCase, height(storedData(testCase)), 2);
verifyStatus(testCase, '전송 중 0');
verifyStatus(testCase, '전송 확인 0');
verifyEqual(testCase, get(control(testCase, 'SensorSend'), 'Enable'), 'off');
press(testCase, 'SensorStop');
sample(testCase, 30000);
verifyStatus(testCase, '보관 2');
verifyEqual(testCase, get(control(testCase, 'SensorSend'), 'Enable'), 'on');
end

function testResumeExcludesPausedTime(testCase)
sample(testCase, 1000);
sample(testCase, 2000);
press(testCase, 'SensorStop');
sample(testCase, 50000);
press(testCase, 'SensorStart');
sample(testCase, 51000);
verifyStatus(testCase, '가위1');
verifyStatus(testCase, '남은 4.0초');
verifyStatus(testCase, '보관 3');
end

function testDeleteLastPreservesEarlierTrial(testCase)
sample(testCase, 1000);
sample(testCase, 9000);
sample(testCase, 10000);
press(testCase, 'SensorStop');
press(testCase, 'SensorDeleteLast');
verifyStatus(testCase, '보관 1');
data = storedData(testCase);
verifyEqual(testCase, height(data), 1);
verifyEqual(testCase, data.source, "가위1");
press(testCase, 'SensorStart');
sample(testCase, 50000);
verifyStatus(testCase, '가위2');
verifyStatus(testCase, '남은 5.0초');
verifyStatus(testCase, '보관 2');
end

function testCompletionStillWaitsForSendAndAllowsRetake(testCase)
for trial = 1:20
    sample(testCase, 1000 + (trial - 1) * 8000);
end
sample(testCase, 161000);
verifyStatus(testCase, '20회 수집 완료');
verifyStatus(testCase, '보관 20');
verifyStatus(testCase, '전송 중 0');
verifyStatus(testCase, '전송 확인 0');
press(testCase, 'SensorDeleteLast');
verifyStatus(testCase, '보관 19');
press(testCase, 'SensorStart');
sample(testCase, 200000);
verifyStatus(testCase, '가위20');
verifyStatus(testCase, '보관 20');
end

function testClearDiscardsMemoryAndUnlocksNewGesture(testCase)
sample(testCase, 1000);
press(testCase, 'SensorStop');
press(testCase, 'SensorClear');
verifyStatus(testCase, '보관 0');
verifyEqual(testCase, height(storedData(testCase)), 0);
verifyEqual(testCase, get(control(testCase, 'SensorGestureName'), 'Enable'), 'on');
verifyEqual(testCase, get(control(testCase, 'SensorSend'), 'Enable'), 'off');
end

function testChangeSubjectAndGestureBeforeSend(testCase)
sample(testCase, 1000);
sample(testCase, 2000);
press(testCase, 'SensorStop');
set(control(testCase, 'SensorSubject'), 'Value', 4); % subject3 (주다경)
set(control(testCase, 'SensorGestureName'), 'String', '못하다');
press(testCase, 'SensorCorrect');
data = storedData(testCase);
verifyEqual(testCase, data.source, ["못하다1"; "못하다1"]);
verifyEqual(testCase, data.subject_id, ["subject3"; "subject3"]);
verifyTrue(testCase, data.session_id(1) ~= "");
verifyTrue(testCase, all(data.session_id == data.session_id(1)));
end

function testCloseKeepsCapturedCsv(testCase)
sample(testCase, 1000);
sample(testCase, 2000);
press(testCase, 'SensorStop');
testCase.TestData.collector.close();
verifyEqual(testCase, height(storedData(testCase)), 2);
end

function testAppendFailureRetainsMemoryForCloseSnapshot(testCase)
sample(testCase, 1000);
path = get(control(testCase, 'SensorCSVFile'), 'TooltipString');
movefile(path, [path '.original']);
mkdir(path); % Force a local write error without relying on permissions.
sample(testCase, 2000);
verifyStatus(testCase, 'CSV 저장 오류');
verifyStatus(testCase, '보관 2');
rmdir(path);
testCase.TestData.collector.close();
verifyEqual(testCase, height(storedData(testCase)), 2);
end

function testCompletedRunReportsCsvDeleteFailure(testCase)
for trial = 1:20
    sample(testCase, 1000 + (trial - 1) * 8000);
end
sample(testCase, 161000);
path = get(control(testCase, 'SensorCSVFile'), 'TooltipString');
movefile(path, [path '.original']);
mkdir(path);
press(testCase, 'SensorDeleteLast');
verifyStatus(testCase, 'CSV 갱신 실패');
verifyStatus(testCase, '보관 20');
rmdir(path);
press(testCase, 'SensorDeleteLast');
verifyEqual(testCase, height(storedData(testCase)), 19);
end

function data = storedData(testCase)
path = get(control(testCase, 'SensorCSVFile'), 'TooltipString');
data = readtable(path, 'FileType', 'text', 'Delimiter', ',', ...
    'TextType', 'string', 'Encoding', 'UTF-8');
end

function sample(testCase, milliseconds)
collector = testCase.TestData.collector;
collector.addSample(milliseconds, 1:6, zeros(1, 6));
collector.tick();
end

function item = control(testCase, tag)
item = findobj(testCase.TestData.fig, 'Tag', tag);
end

function press(testCase, tag)
item = control(testCase, tag);
callback = get(item, 'Callback');
callback(item, []);
end

function verifyStatus(testCase, expected)
verifyTrue(testCase, contains(get(control(testCase, 'SensorStatus'), 'String'), expected));
end
