function tests=test_sign_sequence_collection
% Synthetic frames and deterministic clock only. No BLE/server connections.
tests=functiontests(localfunctions);
end

function testPhaseBoundaries(tc)
p=sign_sequence_phase(4999,20);verifyTrue(tc,p.collecting);verifyEqual(tc,p.step,1);
p=sign_sequence_phase(5000,20);verifyFalse(tc,p.collecting);
p=sign_sequence_phase(8000,20);verifyTrue(tc,p.collecting);verifyEqual(tc,p.step,2);
p=sign_sequence_phase(13000,20);verifyFalse(tc,p.collecting);
p=sign_sequence_phase(16000,20);verifyEqual(tc,p.attempt,2);verifyEqual(tc,p.step,1);
p=sign_sequence_phase(320000,20);verifyTrue(tc,p.done);
end

function testBothHandsAndPairManifest(tc)
folder=tempname;mkdir(folder);time=0;
c=sign_sequence_collection(folder,@currentTime);
cleanup=onCleanup(@() cleanupFiles(c,folder)); %#ok<NASGU>
set(findall(c.figure,'Tag','SequenceCount'),'String','1');
emit(0);
callback=get(findall(c.figure,'Tag','SequenceStart'),'Callback');callback([],[]);
for step=1:2
    for ms=0:100:4900
        time=3+(step-1)*8+ms/1000;emit(ms+(step-1)*8000);
    end
    time=3+(step-1)*8+5;c.tick();
end
time=19;c.tick();
files=dir(fullfile(folder,'seq_*'));experiment=fullfile(files(1).folder,files(1).name);
plan=readtable(fullfile(experiment,'attempt_manifest.csv'),'TextType','string');
verifyEqual(tc,height(plan),2);verifyEqual(tc,plan.step,[1;2]);
verifyEqual(tc,plan.attempt_id(1),plan.attempt_id(2));
verifyEqual(tc,plan.status,["complete";"complete"]);
verifyEqual(tc,plan.data_kind,["synthetic";"synthetic"]);
verifyTrue(tc,all(strlength(plan.capture_start_utc)>0));
verifyEqual(tc,plan.window_start_ms,[0;8000]);
verifyNotEqual(tc,plan.session_id(1),plan.session_id(2));
for k=1:2
    for hand='LR'
        raw=readtable(fullfile(experiment,sprintf('window_%03d_%s.csv',k,hand)),'TextType','string');
        verifyEqual(tc,height(raw),50);verifyEqual(tc,raw.host_elapsed_ms,(0:100:4900)','AbsTol',1e-7);
        verifyEqual(tc,raw.session_id,repmat(plan.session_id(k),50,1));
        verifyEqual(tc,numel(unique(raw.seq)),50);
    end
end
    function emit(ms)
        frame=struct('time_ms',100000+ms,'fsr',1:6,'imu',zeros(1,7));
        for side=1:2,c.addFrame(side,frame);end
    end
    function value=currentTime()
        value=time;
    end
end

function testStopPreservesIncompletePlan(tc)
folder=tempname;mkdir(folder);time=0;
c=sign_sequence_collection(folder,@currentTime);
cleanup=onCleanup(@() cleanupFiles(c,folder)); %#ok<NASGU>
frame=struct('time_ms',100,'fsr',1:6,'imu',zeros(1,7));
for side=1:2,c.addFrame(side,frame);end
callback=get(findall(c.figure,'Tag','SequenceStart'),'Callback');callback([],[]);
time=4;frame.time_ms=200;c.addFrame(1,frame);
callback=get(findall(c.figure,'Tag','SequenceStop'),'Callback');callback([],[]);
files=dir(fullfile(folder,'seq_*'));experiment=fullfile(files(1).folder,files(1).name);
plan=readtable(fullfile(experiment,'attempt_manifest.csv'),'TextType','string');
verifyEqual(tc,height(plan),40);verifyEqual(tc,plan.status(1),"interrupted");
verifyTrue(tc,all(plan.status(2:end)=="not_started"));
raw=readtable(fullfile(experiment,'window_001_L.csv'));
verifyEqual(tc,height(raw),1);
    function value=currentTime()
        value=time;
    end
end

function cleanupFiles(collector,folder)
collector.close();
if isfolder(folder),rmdir(folder,'s');end
end
