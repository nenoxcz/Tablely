# Tablely: 코딩 에이전트용 안내

한 머신에서 여러 학습 작업에 GPU/CPU 코어를 중요도 순으로 나눠주는 스케줄러입니다. 사용자 문서는 README.md(한국어)에 있습니다.

## 커밋

- 커밋 작성자는 저장소 주인 `nenoxcz`입니다. clone마다 한 번 실행하세요:
  `git config user.name nenoxcz && git config user.email 257336585+nenoxcz@users.noreply.github.com`
- 커밋 메시지에 `Co-Authored-By: Claude ...`나 `Claude-Session: ...` 같은 줄은 넣지 않습니다.

## 작업 기록 (자동)

- `.claude/settings.json`의 hooks가 세션마다 `.agents/sessions/<세션ID 앞 8자리>.json`을 자동으로 갱신합니다 (브랜치, 상태, 작업 내용, 수정한 파일).
- 세션을 시작하면 다른 에이전트가 하고 있는 일이 컨텍스트로 들어옵니다. 다른 활성 세션이 고치고 있는 파일은 피하세요.
- 새 작업을 시작할 때 한 줄로 적으세요: `python3 tools/agent_log.py task "<무엇을 하는지>"`
- 자기 세션 파일(`.agents/sessions/...json`)은 작업과 함께 커밋하세요. 다른 세션의 파일은 고치지 마세요.
- 전체 현황: `python3 tools/agent_log.py board --fetch`

## 개발

- 설치와 테스트: `pip install -e '.[test,yaml]' && pytest` (Python 3.10+, 외부 의존성 없음)
- `tablely/planner.py`는 순수 함수입니다. 배분 규칙을 바꾸면 `tests/test_planner.py`부터 고치세요.
- 여러 에이전트 조율은 `tablely/ledger.py`(공용 장부)와 `tablely/runner.py`의 `_tick`이 담당합니다.
- 테스트는 `tests/conftest.py`가 `TABLELY_HOME`을 임시 디렉터리로 바꾸므로 실제 `~/.tablely`를 건드리지 않습니다.
