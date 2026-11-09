import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

from tools import data_upload_report as notifier


def sample(**changes):
    row = dict.fromkeys(notifier.FIELDS, '1')
    row.update(time='2026-11-09T10:00:00', source='REAL', hand='left',
               trial_id='안녕1', label='안녕', subject_id='subject1', session_id='session_1')
    return row | changes


def csv_text(rows, fields=None):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields or notifier.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patcher = patch.object(notifier, 'ROOT', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def inspect(self, rows, fields=None):
        (self.root / 'sample.csv').write_text(csv_text(rows, fields), encoding='utf-8-sig')
        return notifier.inspect_file('sample.csv')

    def test_single_hand_v2_and_legacy_gaps_are_not_false_errors(self):
        result = self.inspect([sample(sample_seq='', ax='')])
        self.assertEqual(result['errors_count'], 0)
        self.assertEqual(result['warnings_count'], 2)
        self.assertEqual(result['hands'], {'left': 1})

    def test_v2_session_can_have_multiple_labels_and_test_rows_are_excluded(self):
        rows = [sample(), sample(sample_seq='2', label='머리', trial_id='머리1'),
                sample(source='TEST', subject_id='', session_id='')]
        result = self.inspect(rows)
        self.assertEqual(result['errors_count'], 0)
        self.assertEqual(result['rows'], 3)
        self.assertEqual(result['valid_rows'], 2)
        self.assertEqual(result['warnings_count'], 1)

    def test_nan_missing_columns_and_duplicate_sample_are_errors(self):
        self.assertGreater(self.inspect([sample(fsr1='nan')])['errors_count'], 0)
        self.assertGreater(self.inspect([sample(fsr2='4096')])['errors_count'], 0)
        result = self.inspect([sample(), sample()])
        self.assertEqual(result['valid_rows'], 1)
        self.assertEqual(result['errors_count'], 1)
        result = self.inspect([{'label': '안녕'}], ['label'])
        self.assertIn('필수 컬럼', result['errors'][0])

    def test_empty_and_truncated_csv_fail(self):
        self.assertGreater(self.inspect([])['errors_count'], 0)
        (self.root / 'bad.csv').write_text(csv_text([]) + 'only,three,columns\n', encoding='utf-8')
        self.assertGreater(notifier.inspect_file('bad.csv')['errors_count'], 0)

    def test_v3_timing_and_paired_hand_warning(self):
        left = sample(gesture_type='left_only', host_elapsed_ms='100', trial_elapsed_ms='100')
        fields = notifier.FIELDS + notifier.TIMED_FIELDS
        result = self.inspect([left], fields)
        self.assertEqual(result['warnings_count'], 1)
        right = left | {'hand': 'right'}
        result = self.inspect([left, right], fields)
        self.assertEqual(result['errors_count'], 0)
        self.assertEqual(result['warnings_count'], 0)
        result = self.inspect([left | {'host_elapsed_ms': '8000'}], fields)
        self.assertGreater(result['errors_count'], 0)

    def make_zip(self, tamper=False, bad_count=False):
        data = csv_text([sample()]).encode('utf-8-sig')
        subset = data  # Same rows must not be counted a second time.
        manifest = {'real_rows': 2 if bad_count else 1, 'session_count': 1, 'files': {
            'data/sensor_dataset_v2.csv': hashlib.sha256(data).hexdigest(),
            'XIAO/day.csv': hashlib.sha256(subset).hexdigest()}}
        with zipfile.ZipFile(self.root / 'snapshot.zip', 'w') as archive:
            archive.writestr('package/data/sensor_dataset_v2.csv', data)
            archive.writestr('package/XIAO/day.csv', b'changed' if tamper else subset)
            archive.writestr('package/manifest.json', json.dumps(manifest))

    def test_zip_checks_integrity_and_counts_only_master(self):
        self.make_zip()
        result = notifier.inspect_file('snapshot.zip')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['rows'], 1)
        self.make_zip(tamper=True)
        self.assertIn('SHA-256', notifier.inspect_file('snapshot.zip')['errors'][0])
        self.make_zip(bad_count=True)
        self.assertGreater(notifier.inspect_file('snapshot.zip')['errors_count'], 0)

    def test_unsafe_zip_and_deleted_file(self):
        with zipfile.ZipFile(self.root / 'unsafe.zip', 'w') as archive:
            archive.writestr('../outside.csv', b'bad')
        self.assertGreater(notifier.inspect_file('unsafe.zip')['errors_count'], 0)
        deleted = notifier.inspect_file('gone.csv', 'D')
        self.assertEqual(deleted['errors_count'], 0)
        self.assertEqual(deleted['warnings_count'], 1)

    def test_push_range_includes_deletion_and_ignores_model_outputs(self):
        event = self.root / 'event.json'
        event.write_text(json.dumps({'before': 'a' * 40, 'after': 'b' * 40}), encoding='utf-8')
        diff = 'M\0data/sensor_dataset_v2.csv\0D\0datasets/old.zip\0M\0data/knn_results/a.csv\0'
        with patch.dict(os.environ, GITHUB_EVENT_NAME='push', GITHUB_EVENT_PATH=str(event)):
            with patch.object(notifier, 'git', return_value=diff) as command:
                files = notifier.changed_files()
                self.assertIn(('datasets/old.zip', 'D'), files)
                self.assertEqual(len(files), 2)
                self.assertEqual(command.call_args.args[-2:], ('a' * 40, 'b' * 40))


class DiscordTests(unittest.TestCase):
    def report(self):
        result = dict(path='datasets/@everyone.zip', change='A', rows=1, valid_rows=1,
                      hands={'left': 1}, labels={'안녕': 1}, sessions=1, trials=1,
                      errors=[], warnings=[], errors_count=0, warnings_count=0)
        return dict(actor='teammate', branch='main', run_url='', files=[result])

    def test_payload_disables_mentions_and_fits_discord_limits(self):
        report = self.report()
        report['files'] *= 100
        message = notifier.payload(report)
        self.assertEqual(message['allowed_mentions'], {'parse': []})
        self.assertNotIn('@everyone', message['embeds'][0]['description'])
        self.assertLessEqual(len(message['embeds'][0]['description']), 4096)

    def test_missing_secret_and_no_changes_never_send(self):
        with patch.dict(os.environ, DISCORD_WEBHOOK_URL=''):
            with patch.object(notifier.urllib.request, 'urlopen') as request:
                notifier.send_discord(self.report())
                notifier.send_discord({'files': []})
                request.assert_not_called()

    def test_rate_limit_retries_and_waits_for_delivery_response(self):
        url = 'https://discord.com/api/webhooks/123/fake-token'
        error = urllib.error.HTTPError(url, 429, 'rate limited', {}, io.BytesIO(b'{"retry_after": 0.1}'))
        with patch.dict(os.environ, DISCORD_WEBHOOK_URL=url):
            with patch.object(notifier.urllib.request, 'urlopen', side_effect=[error, unittest.mock.MagicMock()]) as request:
                with patch.object(notifier.time, 'sleep'):
                    notifier.send_discord(self.report())
                self.assertEqual(request.call_count, 2)
                self.assertTrue(request.call_args.args[0].full_url.endswith('?wait=true'))

    def test_http_error_does_not_expose_webhook_secret(self):
        url = 'https://discord.com/api/webhooks/123/fake-token'
        error = urllib.error.HTTPError(url, 401, 'unauthorized', {}, None)
        with patch.dict(os.environ, DISCORD_WEBHOOK_URL=url):
            with patch.object(notifier.urllib.request, 'urlopen', side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    notifier.send_discord(self.report())
                self.assertNotIn('fake-token', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
