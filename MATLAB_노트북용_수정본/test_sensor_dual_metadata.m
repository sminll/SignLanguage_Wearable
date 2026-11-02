function tests=test_sensor_dual_metadata
tests=functiontests(localfunctions);
end

function testLegacyPayloadUnchanged(testCase)
p=make_sensor_payload(1:6,zeros(1,6),'L',1,100,'가위1','subject1','test_session');
verifyFalse(testCase,isfield(p,'gesture_type'));
end

function testTimedCsvAndSchemaGuard(testCase)
folder=tempname; mkdir(folder);
cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
file=fullfile(folder,'timed.csv');
write_sensor_csv(file,{},'replace',3);
p=make_sensor_payload(1:6,zeros(1,6),'R',1,90000,'가위1', ...
    'subject1','test_session','left_only',100,100);
write_sensor_csv(file,{p},'append');
t=readtable(file,'TextType','string');
verifyEqual(testCase,width(t),22);
verifyEqual(testCase,t.gesture_type,"left_only");
verifyEqual(testCase,t.trial_elapsed_ms,100);
legacy=make_sensor_payload(1:6,zeros(1,6),'R',2,90100,'가위1');
verifyError(testCase,@() write_sensor_csv(file,{legacy},'append'),'sensor:CsvSchemaMismatch');
verifyEqual(testCase,height(readtable(file)),1);
end

function testThreeTypesCaptureBothHands(testCase)
% Real monitor/decoder/CSV code, simulated BLE only. No server requests.
root=fileparts(mfilename('fullpath'));
previous=pwd; oldPath=path; folder=tempname; mkdir(folder);
files={'xiao_ble_dual_monitor.m','xiao_ble_packet_decode.m', ...
    'make_sensor_payload.m','write_sensor_csv.m','post_sensor_batch.m','sensor_collection_phase.m'};
for k=1:numel(files), copyfile(fullfile(root,files{k}),folder); end
copyfile(fullfile(root,'tests/support/FakeBLECharacteristic.m'),folder);
writeStub(folder,'ble','function b=ble(address)\nb=address;\nend\n');
writeStub(folder,'characteristic', ...
    ['function c=characteristic(address,varargin)\n' ...
     'global XIAO_DUAL_MOCK; if strcmp(address,''LEFT''), c=XIAO_DUAL_MOCK{1}; else, c=XIAO_DUAL_MOCK{2}; end\nend\n']);
writeStub(folder,'subscribe','function subscribe(varargin)\nend\n');
writeStub(folder,'unsubscribe','function unsubscribe(varargin)\nend\n');
writeStub(folder,'questdlg','function answer=questdlg(varargin)\nanswer=varargin{3};\nend\n');
addpath(folder,'-begin'); cd(folder);
clear ble characteristic subscribe unsubscribe questdlg xiao_ble_dual_monitor;
global XIAO_DUAL_MOCK;
XIAO_DUAL_MOCK={FakeBLECharacteristic,FakeBLECharacteristic};
failure=[]; done=false; started=tic;
driver=timer('ExecutionMode','fixedSpacing','Period',0.5,'BusyMode','drop','TimerFcn',@drive);
cleanup=onCleanup(@restore); %#ok<NASGU>
start(driver);
xiao_ble_dual_monitor('LEFT','RIGHT');
if ~isempty(failure), clear cleanup; rethrow(failure); end
verifyTrue(testCase,done);
clear cleanup;

    function drive(~,~)
        f=findall(groot,'Tag','XIAODualBLE');
        if isempty(f),return;end
        if toc(started)>120
            failure=MException('test:Timeout','Mock dual capture timeout');delete(f);return;
        end
        if isempty(XIAO_DUAL_MOCK{1}.DataAvailableFcn) || isempty(XIAO_DUAL_MOCK{2}.DataAvailableFcn),return;end
        try
            set(findall(f,'Style','edit'),'String','가위');
            popups=findall(f,'Style','popupmenu');
            for popup=popups'
                names=get(popup,'String');
                if contains(string(names{1}),'측정자'),set(popup,'Value',2);end
            end
            types={'left_only','right_only','both'};
            for mode=1:3
                set(findall(f,'Tag','GestureType'),'Value',mode);
                for side=1:2,XIAO_DUAL_MOCK{side}.emit(mode*2,mode*200);end
                callback=get(findall(f,'String','양손 20회 수집 시작'),'Callback');callback([],[]);
                verifyEqual(testCase,get(findall(f,'Tag','GestureType'),'Enable'),'off');
                for side=1:2,XIAO_DUAL_MOCK{side}.emit(mode*2+1,mode*200+50);end
                callback=get(findall(f,'String','일시 중지'),'Callback');callback([],[]);
                csvs=dir(fullfile(folder,'sensor_data','*.csv'));
                verifyEqual(testCase,numel(csvs),mode*2);
                newest=cell(1,2);
                for side=1:2
                    hand='L';if side==2,hand='R';end
                    found=dir(fullfile(folder,'sensor_data',['*_' hand '_*.csv']));
                    for idx=1:numel(found)
                        candidate=readtable(fullfile(found(idx).folder,found(idx).name),'TextType','string');
                        if height(candidate)==1 && candidate.gesture_type==string(types{mode})
                            newest{side}=candidate;
                        end
                    end
                    verifyEqual(testCase,height(newest{side}),1);
                    verifyEqual(testCase,newest{side}.gesture_type,string(types{mode}));
                    verifyGreaterThanOrEqual(testCase,newest{side}.trial_elapsed_ms,0);
                end
                verifyEqual(testCase,newest{1}.session_id,newest{2}.session_id);
                verifyEqual(testCase,newest{1}.source,newest{2}.source);
                callback=get(findall(f,'String','새 수집/초기화'),'Callback');callback([],[]);
            end
            done=true;delete(f);
        catch err
            failure=err;delete(f);
        end
    end

    function restore()
        stop(driver);delete(driver);
        f=findall(groot,'Tag','XIAODualBLE');if ~isempty(f),delete(f);end
        cd(previous);path(oldPath);
        clear ble characteristic subscribe unsubscribe questdlg xiao_ble_dual_monitor;
        clear global XIAO_DUAL_MOCK;
        rmdir(folder,'s');
    end
end

function writeStub(folder,name,source)
fid=fopen(fullfile(folder,[name '.m']),'w');
fprintf(fid,source);fclose(fid);
end
