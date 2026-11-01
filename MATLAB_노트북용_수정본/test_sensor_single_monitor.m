function tests=test_sensor_single_monitor
% Disposable BLE packets verify capture/UI/CSV without a physical board.
tests=functiontests(localfunctions);
end

function setupOnce(testCase)
root=fileparts(mfilename('fullpath'));
testCase.TestData.oldpath=path;
folder=tempname;mkdir(folder);
testCase.TestData.folder=folder;
copyfile(fullfile(root,'xiao_ble_single_ui_test.m'),folder);
addpath(fullfile(root,'tests/support'));
addpath(folder,'-begin');
clear xiao_ble_single_ui_test;
end

function teardownOnce(testCase)
path(testCase.TestData.oldpath);
clear xiao_ble_single_ui_test;
rmdir(testCase.TestData.folder,'s');
end

function testCapture(testCase)
c=FakeBLECharacteristic;
xiao_ble_single_ui_test(c);
f=findall(groot,'Type','figure','Name','XIAO BLE 양손 센서 모니터 (1개 보드 실시간 테스트)');
assert(isscalar(f));
cleanup=onCleanup(@() close(f(isgraphics(f))));
c.emit(1,100);
label=findall(f,'Style','edit');set(label,'String','시험');
popups=findall(f,'Style','popupmenu');
for p=popups'
    choices=get(p,'String');
    if contains(string(choices{1}),'측정자'), set(p,'Value',7); end
    if strcmp(choices{1},'왼손'), set(p,'Value',2); end
end
button=findall(f,'Style','pushbutton','String','20회 수집 시작');
c.emit(1,100);
cb=get(button,'Callback'); cb([],[]);
for k=2:6, c.emit(k,k*100); end
button=findall(f,'Style','pushbutton','String','일시 중지'); cb=get(button,'Callback'); cb([],[]);
button=findall(f,'Style','pushbutton','String','Python 서버로 전송');
if ~strcmp(get(button,'Enable'),'on')
    texts=findall(f,'Style','text');
    for text=texts', disp(get(text,'String')); end
end
assert(strcmp(get(button,'Enable'),'on'));
folder=fullfile(fileparts(which('xiao_ble_single_ui_test')),'sensor_data');
files=dir(fullfile(folder,'*.csv'));[~,i]=max([files.datenum]);
data=readtable(fullfile(folder,files(i).name),'TextType','string');
assert(height(data)==5);
assert(all(data.device_id=="R"));
assert(all(data.subject_id=="subject6"));
assert(all(data.source=="시험1"));
assert(isequal(data.seq,(1:5)'));
assert(all(data.fsr1_raw==100));
close(f);clear cleanup;
assert(isempty(c.DataAvailableFcn));
assert(isempty(findall(groot,'Type','figure','Name','XIAO BLE 양손 센서 모니터 (1개 보드 실시간 테스트)')));
verifyTrue(testCase,true);
end

function testOldCurrentFolderDoesNotShadowPayload(testCase)
previous=pwd;
shadow=tempname; mkdir(shadow);
fid=fopen(fullfile(shadow,'make_sensor_payload.m'),'w');
fprintf(fid,'function p=make_sensor_payload(a,b,c,d,e,f)\np=struct();\nend\n');
fclose(fid);
cleanup=onCleanup(@() restoreShadow(previous,shadow)); %#ok<NASGU>
cd(shadow);
assert(nargin('make_sensor_payload')==6);
testCapture(testCase);
verifyEqual(testCase,pwd,shadow);
end

function restoreShadow(previous,shadow)
cd(previous);
rmdir(shadow,'s');
end
