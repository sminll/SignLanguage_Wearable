import copy
import io
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import Mock, patch

from tools import data_notification_digest as digest
from tools.notification_store import Discord, GitHubStore, HttpFailure
from test_data_upload_report import csv_text, sample


CONFIG = {'default_target': 20, 'quiet_seconds': 60,
          'test_label_contains': ['테스트', '디스코드']}


def uploaded(label, trial, *, hand='left', seq='1', fsr='1'):
    row = sample(label=label, trial_id=trial, hand=hand, sample_seq=seq,
                 fsr1=fsr, time='2026-11-09T10:00:00Z')
    name = f'sensor-data/training/user/{label}-{trial}-{hand}.csv'
    return name, {'blob': trial, **digest.inspect_blob(name, csv_text([row]).encode())}


class GroupingTests(unittest.TestCase):
    def test_inventory_uses_available_roots_and_caches_unchanged_files(self):
        path = 'sensor-data/training/user/upload.csv'
        blob = csv_text([sample()]).encode()
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as archive:
            member = tarfile.TarInfo(path)
            member.size = len(blob)
            archive.addfile(member, io.BytesIO(blob))
        tree = f'100644 blob abc123\t{path}\0'.encode()
        with patch.object(digest, 'git', side_effect=[tree, stream.getvalue()]) as git:
            files, changed = digest.inventory('new-head', {})
            self.assertEqual(git.call_args.args, ('archive', 'new-head', 'sensor-data'))
            self.assertEqual(changed, {path})
        with patch.object(digest, 'git', return_value=tree) as git:
            self.assertEqual(digest.inventory('same-files', files), (files, set()))
            self.assertEqual(git.call_count, 1)

    def test_60_files_are_one_participant_card_with_three_20_trial_rows(self):
        files = dict(uploaded(label, f'trial-{n}') for label in ('삼키다', '상처', '교통사고') for n in range(20))
        cards = digest.group_cards(files, CONFIG)
        self.assertEqual(len(cards), 1)
        card = next(iter(cards.values()))
        self.assertEqual(len(card['labels']), 3)
        self.assertTrue(all(row['uploaded'] == row['passed'] == 20 for row in card['labels']))
        self.assertTrue(all(row['status'] == '완료' for row in card['labels']))

    def test_hands_are_one_trial_and_resends_do_not_inflate_counts(self):
        name, file = uploaded('머리', 'one')
        row = sample(label='머리', trial_id='one', hand='right', time='2026-11-09T10:00:00Z')
        left = sample(label='머리', trial_id='one', hand='left', time='2026-11-09T10:00:00Z')
        file = digest.inspect_blob(name, csv_text([left, row]).encode())
        cards = digest.group_cards({name: file, 'resend.csv': copy.deepcopy(file)}, CONFIG)
        result = next(iter(cards.values()))['labels'][0]
        self.assertEqual(result['uploaded'], 1)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(result['errors'], 0)

    def test_conflicting_duplicate_is_one_failed_trial(self):
        name, first = uploaded('머리', 'one')
        _, second = uploaded('머리', 'one', fsr='2')
        cards = digest.group_cards({name: first, 'another.csv': second}, CONFIG)
        result = next(iter(cards.values()))['labels'][0]
        self.assertEqual(result['uploaded'], 1)
        self.assertEqual(result['errors'], 1)
        self.assertEqual(result['passed'], 0)

    def test_people_sessions_and_korean_dates_are_separate(self):
        name, first = uploaded('머리', 'one')
        second = copy.deepcopy(first)
        second['records'][0]['subject'] = 'subject2'
        third = copy.deepcopy(first)
        third['records'][0]['session'] = 'another-session'
        self.assertEqual(len(digest.group_cards({name: first, 'second': second, 'third': third}, CONFIG)), 3)
        self.assertEqual(digest.capture_day('2026-11-09T15:01:00Z'), '2026-11-10')
        self.assertEqual(digest.capture_day('2026-11-09T20:00:00'), '2026-11-09')

    def test_trial_spanning_midnight_is_not_counted_twice(self):
        rows = [sample(time='2026-11-09T14:59:59Z'),
                sample(time='2026-11-09T15:00:01Z', sample_seq='2')]
        file = digest.inspect_blob('sensor-data/training/user/one.csv', csv_text(rows).encode())
        self.assertEqual(len(file['records']), 1)
        self.assertEqual(file['records'][0]['day'], '2026-11-09')

    def test_missing_failed_and_test_capture_targets(self):
        files = dict(uploaded('머리', str(n), fsr='nan' if n == 18 else '1') for n in range(19))
        row = next(iter(digest.group_cards(files, CONFIG).values()))['labels'][0]
        self.assertEqual((row['uploaded'], row['passed'], row['errors'], row['missing']), (19, 18, 1, 1))
        self.assertEqual(digest.expected('디스코드 테스트', CONFIG), 1)
        self.assertEqual(digest.expected('머리', CONFIG | {'targets': {'머리': 10}}), 10)

    def test_broken_csv_is_visible_and_not_silently_dropped(self):
        file = digest.inspect_blob('sensor-data/training/user/bad.csv', b'bad,header\n1,2\n')
        card = next(iter(digest.group_cards({'bad': file}, CONFIG).values()))
        self.assertGreater(card['labels'][0]['errors'], 0)

    def test_modified_and_deleted_files_refresh_the_old_card(self):
        name, file = uploaded('머리', 'one')
        keys = digest.affected_keys({name}, {name: file}, {})
        self.assertEqual(len(keys), 1)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        name, file = uploaded('머리', 'one')
        self.cards = digest.group_cards({name: file}, CONFIG)
        self.key = next(iter(self.cards))
        self.state = {'head': 'previous', 'files': {name: file}, 'messages': {}, 'pending_keys': [self.key]}
        self.store = Mock()
        self.discord = Mock()
        self.discord.deliver.return_value = '123'

    def deliver(self):
        return digest.deliver_cards(self.cards, {self.key}, self.state, self.store, self.discord,
                                    CONFIG, 'owner/repo', 'training-data', 'https://github.com/owner/repo/actions')

    def test_new_card_once_updates_edit_and_identical_rerun_is_noop(self):
        self.assertEqual(self.deliver(), 1)
        self.assertIsNone(self.discord.deliver.call_args.args[1])
        self.assertEqual(self.state['pending_keys'], [])
        self.assertEqual(self.deliver(), 0)
        self.cards[self.key]['labels'][0]['uploaded'] = 2
        self.assertEqual(self.deliver(), 1)
        self.assertEqual(self.discord.deliver.call_args.args[1], '123')
        self.assertEqual(self.state['head'], 'previous')  # Only the final transaction advances the baseline.

    def test_failed_delivery_remains_pending_for_retry(self):
        self.discord.deliver.side_effect = RuntimeError('network')
        with self.assertRaises(RuntimeError):
            self.deliver()
        self.assertEqual(self.state['pending_keys'], [self.key])
        self.assertEqual(self.state['messages'], {})

    def test_partial_failure_keeps_remaining_people_and_does_not_resend_first(self):
        second = copy.deepcopy(self.cards[self.key])
        second['subject'] = 'subject2'
        other_key = digest.card_key(second['subject'], second['session'], second['day'])
        self.cards[other_key] = second
        keys = set(self.cards)
        self.state['pending_keys'] = sorted(keys)
        self.discord.deliver.side_effect = ['first-id', RuntimeError('network')]
        with self.assertRaises(RuntimeError):
            digest.deliver_cards(self.cards, keys, self.state, self.store, self.discord,
                                 CONFIG, 'owner/repo', 'training-data', 'https://github.com/owner/repo/actions')
        self.assertEqual(len(self.state['pending_keys']), 1)
        self.assertEqual(len(self.state['messages']), 1)
        self.discord.deliver.side_effect = None
        digest.deliver_cards(self.cards, keys, self.state, self.store, self.discord,
                             CONFIG, 'owner/repo', 'training-data', 'https://github.com/owner/repo/actions')
        self.assertEqual(self.discord.deliver.call_count, 3)
        self.assertEqual(self.state['pending_keys'], [])

    def test_existing_card_deletion_becomes_a_warning_update(self):
        self.deliver()
        self.cards = {}
        self.deliver()
        self.assertIn('삭제', self.discord.deliver.call_args.args[0]['embeds'][0]['title'])

    def test_deleting_bootstrapped_history_also_emits_a_warning(self):
        self.cards = {}
        self.assertEqual(self.deliver(), 1)
        self.assertIn('삭제', self.discord.deliver.call_args.args[0]['embeds'][0]['title'])

    def test_existing_webhook_message_uses_patch_and_does_not_ping(self):
        client = Discord('https://discord.com/api/webhooks/123/fake-token')
        with patch('tools.notification_store.request_json', return_value={'id': '42'}) as request:
            self.assertEqual(client.deliver({'embeds': []}, '42'), '42')
            self.assertEqual(request.call_args.kwargs['method'], 'PATCH')
            self.assertEqual(request.call_args.kwargs['body']['allowed_mentions'], {'parse': []})

    def test_deleted_discord_card_is_recreated_silently(self):
        client = Discord('https://discord.com/api/webhooks/123/fake-token')
        with patch('tools.notification_store.request_json', side_effect=[HttpFailure('Discord', 404), {'id': '43'}]) as request:
            self.assertEqual(client.deliver({'embeds': []}, '42'), '43')
            self.assertEqual(request.call_args.kwargs['body']['flags'], 4096)

    def test_quiet_window_restarts_on_new_uploads(self):
        now = [0]
        def sleep(n): now[0] += n
        store = Mock()
        store.latest.side_effect = ['one', 'two', 'two', 'two']
        head = digest.wait_for_quiet(store, 20, sleep=sleep, clock=lambda: now[0])
        self.assertEqual(head, 'two')
        self.assertEqual(now[0], 30)

    def test_model_reference_is_never_claimed_as_new_upload_accuracy(self):
        config = CONFIG | {'model_reference': {'labels': ['삼키다'], 'correct': 413, 'trials': 559,
                                               'evaluated_on': '2026-11-08', 'source_url': 'https://example.com/evaluation'}}
        text = digest.model_reference(config, ['삼키다'])
        self.assertIn('73.88%', text)
        self.assertIn('이전 내부 교차검증', text)
        self.assertIn('미평가', text)
        self.assertNotIn('73.88%', digest.model_reference(config, ['머리']))

    def test_manual_check_does_not_swallow_another_persons_new_upload(self):
        first_name, first = uploaded('머리', 'one')
        second_name, second = uploaded('삼키다', 'two')
        second['records'][0]['subject'] = 'subject2'
        before = {first_name: first}
        after = before | {second_name: second}
        state = {'version': 2, 'head': 'old', 'files': before, 'messages': {}}
        store = Mock()
        store.load.return_value = state
        store.latest.return_value = 'new'
        discord = Mock()
        discord.deliver.return_value = 'message-id'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'settings.json'
            config.write_text(json.dumps(CONFIG), encoding='utf-8')
            with patch.object(digest, 'CONFIG', config), patch.object(digest, 'GitHubStore', return_value=store), \
                 patch.object(digest, 'Discord', return_value=discord), patch.object(digest, 'git', return_value=b'new'), \
                 patch.object(digest, 'inventory', return_value=(after, {second_name})), \
                 patch.dict(os.environ, GITHUB_EVENT_NAME='workflow_dispatch', DATASET_PATH=first_name,
                            GITHUB_TOKEN='fake', DISCORD_WEBHOOK_URL='fake', BOOTSTRAP='false'), \
                 patch('sys.argv', ['digest.py', '--report', str(root/'report.json')]):
                self.assertEqual(digest.main(), 0)
            self.assertEqual(discord.deliver.call_count, 2)
            self.assertEqual(len(state['messages']), 2)

    def test_large_state_uses_blob_api_when_contents_api_omits_content(self):
        import base64
        store = GitHubStore('owner/repo', 'training-data', 'fake-token')
        state = {'version': 2, 'head': 'abc', 'files': {}, 'messages': {}}
        encoded = base64.b64encode(json.dumps(state).encode()).decode()
        store.api = Mock(side_effect=[{'sha': 'state-sha', 'encoding': 'none', 'content': ''},
                                      {'encoding': 'base64', 'content': encoded}])
        self.assertEqual(store.load(), state)
        self.assertEqual(store.api.call_args.args[0], '/git/blobs/state-sha')


if __name__ == '__main__':
    unittest.main()
