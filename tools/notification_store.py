"""Small GitHub state store and Discord transport; credentials stay in memory."""
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


class HttpFailure(RuntimeError):
    def __init__(self, service, status):
        self.status = status
        super().__init__(f'{service}: HTTP {status}')


def request_json(url, *, method='GET', body=None, headers=None, service='API'):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None
    for attempt in range(4):
        request = urllib.request.Request(url, data=data, method=method, headers={
            'Content-Type': 'application/json', 'User-Agent': 'Wearable-Notification/2', **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                if status == 429 and attempt < 3:
                    try:
                        delay = float(json.loads(exc.read()).get('retry_after', 2))
                    except (ValueError, TypeError):
                        delay = 2
                    time.sleep(min(max(delay, 1), 30))
                    continue
            finally:
                exc.close()
            raise HttpFailure(service, status) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError(f'{service}: 연결 실패 또는 응답 시간 초과') from None


class GitHubStore:
    """One JSON file per source branch, in a separate state branch."""
    def __init__(self, repo, branch, token):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('저장소 이름이 잘못됐습니다.')
        self.base = f'https://api.github.com/repos/{repo}'
        self.branch = branch
        self.token = token
        self.headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                        'X-GitHub-Api-Version': '2022-11-28'}
        self.state_branch = 'discord-notification-state'
        self.path = 'notifications/' + hashlib.sha256(branch.encode()).hexdigest()[:16] + '.json'
        self.sha = None

    def api(self, path, *, method='GET', body=None):
        return request_json(self.base + path, method=method, body=body, headers=self.headers, service='GitHub')

    def latest(self):
        ref = urllib.parse.quote(self.branch, safe='')
        return self.api('/commits/' + ref)['sha']

    def load(self):
        try:
            data = self.api('/contents/' + self.path + '?ref=' + self.state_branch)
        except HttpFailure as exc:
            if exc.status == 404:
                return {'version': 2, 'head': '', 'files': {}, 'messages': {}}
            raise
        self.sha = data['sha']
        if data.get('encoding') != 'base64' or not data.get('content'):
            data = self.api('/git/blobs/' + self.sha)
        state = json.loads(base64.b64decode(data['content']))
        if state.get('version') != 2:
            raise ValueError('알림 상태 버전이 다릅니다. 상태를 자동 삭제하지 않습니다.')
        return state

    def save(self, state):
        try:
            self.api('/git/ref/heads/' + self.state_branch)
        except HttpFailure as exc:
            if exc.status != 404:
                raise
            default = self.api('')['default_branch']
            seed = self.api('/commits/' + urllib.parse.quote(default, safe=''))['sha']
            try:
                self.api('/git/refs', method='POST', body={'ref': 'refs/heads/' + self.state_branch, 'sha': seed})
            except HttpFailure as create_error:
                if create_error.status != 422:  # Another source branch may have created it.
                    raise
        content = base64.b64encode(json.dumps(state, ensure_ascii=False, separators=(',', ':')).encode()).decode()
        for attempt in range(4):
            body = {'message': 'Update Discord notification state', 'content': content, 'branch': self.state_branch}
            if self.sha:
                body['sha'] = self.sha
            try:
                result = self.api('/contents/' + self.path, method='PUT', body=body)
                self.sha = result['content']['sha']
                return
            except HttpFailure as exc:
                if exc.status != 409 or attempt == 3:
                    raise
                current = self.api('/contents/' + self.path + '?ref=' + self.state_branch)
                self.sha = current['sha']
                time.sleep(1)

    def git_environment(self):
        env = dict(os.environ)
        auth = base64.b64encode(('x-access-token:' + self.token).encode()).decode()
        env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + auth, GIT_TERMINAL_PROMPT='0')
        return env


class Discord:
    def __init__(self, url):
        parts = urllib.parse.urlsplit(url.strip())
        if (parts.scheme != 'https' or parts.netloc != 'discord.com'
                or not re.fullmatch(r'/api(?:/v\d+)?/webhooks/\d+/[A-Za-z0-9._-]+', parts.path)):
            raise ValueError('공식 Discord 웹훅 주소가 필요합니다.')
        self.base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))

    def deliver(self, payload, message_id=None, *, silent=False):
        body = dict(payload, allowed_mentions={'parse': []})
        if message_id:
            try:
                return request_json(self.base + '/messages/' + str(message_id), method='PATCH',
                                    body=body, service='Discord')['id']
            except HttpFailure as exc:
                if exc.status != 404:
                    raise
                silent = True  # A deleted card is restored without another push notification.
        if silent:
            body['flags'] = 4096  # SUPPRESS_NOTIFICATIONS
        return request_json(self.base + '?wait=true', method='POST', body=body, service='Discord')['id']
