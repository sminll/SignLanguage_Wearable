function tests=test_sensor_upload
% Integration checks against an explicitly configured local disposable receiver.
tests=functiontests(localfunctions);
end

function testSuccessfulPost(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'), ...
    'Set XIAO_TEST_RECEIVER to a disposable localhost receiver.');
r=post_sensor_batch([url '/sensor'],makeBatch());
verifyEqual(testCase,r.sent,3);
verifyEqual(testCase,r.unconfirmed,0);
verifyEmpty(testCase,r.message);
end

function testStopsAtFirstUnconfirmedPost(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'), ...
    'Set XIAO_TEST_RECEIVER to a disposable localhost receiver.');
% /fail must acknowledge seq=1 and return HTTP 503 for seq=2.
r=post_sensor_batch([url '/fail'],makeBatch());
verifyEqual(testCase,r.sent,1);
verifyEqual(testCase,r.unconfirmed,2);
verifyNotEmpty(testCase,r.message);
end

function testBackgroundPost(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'), ...
    'Set XIAO_TEST_RECEIVER to a disposable localhost receiver.');
future=parfeval(backgroundPool,@post_sensor_batch,1,[url '/sensor'],makeBatch());
cleanup=onCleanup(@() delete(future)); %#ok<NASGU>
r=fetchOutputs(future);
verifyEqual(testCase,r.sent,3);
verifyEqual(testCase,r.unconfirmed,0);
end

function batch=makeBatch()
batch=cell(1,3);
for k=1:3
    batch{k}=make_sensor_payload(1:6,[0 0 1 2 3 4],'R',k,100*k, ...
        '시험1','subject6','local_integration_test');
end
end

function testIdempotentServer500Recovery(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'));
batch=makeBatch();
for k=1:numel(batch), batch{k}.session_id='retry_server_error_test'; end
r=post_sensor_batch([url '/saved-then-500'],batch,true);
verifyEqual(testCase,r.sent,3);
verifyEqual(testCase,r.unconfirmed,0);
verifyEmpty(testCase,r.message);
end

function testRepeated500StillStops(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'));
r=post_sensor_batch([url '/always-500'],makeBatch(),true);
verifyEqual(testCase,r.sent,0);
verifyEqual(testCase,r.unconfirmed,3);
verifyNotEmpty(testCase,r.message);
end

function testDefaultDoesNotRetry500(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'));
batch=makeBatch();
for k=1:numel(batch), batch{k}.session_id='no_retry_server_error_test'; end
r=post_sensor_batch([url '/saved-then-500'],batch);
verifyEqual(testCase,r.sent,0);
verifyEqual(testCase,r.unconfirmed,3);
end

function testTimedMetadataPost(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'));
p=make_sensor_payload(1:6,zeros(1,6),'R',1,90000,'시험1', ...
    'subject1','timed_metadata_test','left_only',100,100);
r=post_sensor_batch([url '/sensor'],{p},true);
verifyEqual(testCase,r.sent,1);
verifyEmpty(testCase,r.message);
end

function testOldServerCannotSilentlyDropMetadata(testCase)
url=getenv('XIAO_TEST_RECEIVER');
assumeTrue(testCase,startsWith(url,'http://127.0.0.1:'));
p=make_sensor_payload(1:6,zeros(1,6),'L',1,100,'시험1', ...
    'subject1','old_server_metadata_test','both',100,100);
r=post_sensor_batch([url '/legacy/sensor'],{p},true);
verifyEqual(testCase,r.sent,0);
verifyEqual(testCase,r.unconfirmed,1);
verifyNotEmpty(testCase,r.message);
end
