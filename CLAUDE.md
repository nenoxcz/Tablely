# Tablely: 코딩 에이전트용 안내

한 머신에서 여러 학습 작업에 GPU/CPU 코어를 중요도 순으로 나눠주는 스케줄러입니다. 사용자 문서는 README.md(한국어)에 있습니다.

## 커밋

- 커밋 작성자는 저장소 주인 `nenoxcz`입니다. clone마다 한 번 실행하세요:
  `git config user.name nenoxcz && git config user.email 257336585+nenoxcz@users.noreply.github.com`
- 커밋 메시지에 `Co-Authored-By: Claude ...`나 `Claude-Session: ...` 같은 줄은 넣지 않습니다.

## 작업 기록 (자동)

- `.claude/settings.json`의 hooks가 세션마다 작업 기록을 자동으로 갱신합니다 (브랜치, 상태, 작업 내용, 수정한 파일). 실시간 상태는 `.git/agent-sessions/`(커밋 안 됨)에, 커밋할 사본은 `.agents/sessions/<세션ID 앞 8자리>.json`에 있습니다. 커밋할 사본은 파일을 수정할 때만 바뀝니다.
- 세션을 시작하면 핸드오프가 컨텍스트로 들어옵니다. 최근 세션들이 어디서 멈췄는지와 지금 누가 작업 중인지입니다. 이어서 할 일이 있으면 거기서부터 시작하고, 다른 활성 세션이 고치고 있는 파일은 피하세요.
- 새 작업을 시작할 때 한 줄로 적으세요: `python3 tools/agent_log.py task "<무엇을 하는지>"`
- 작업을 마치기 전, 마지막 커밋 전에 핸드오프를 남기세요: `python3 tools/agent_log.py handoff "<끝낸 것 / 다음 할 것 / 주의할 점>"`
- 자기 세션 파일(`.agents/sessions/...json`)은 작업과 함께 커밋하세요. 다른 세션의 파일은 고치지 마세요.
- 전체 현황: `python3 tools/agent_log.py board --fetch`

## 개발

- 설치와 테스트: `pip install -e '.[test,yaml]' && pytest` (Python 3.10+, 외부 의존성 없음)
- `tablely/planner.py`는 순수 함수입니다. 배분 규칙을 바꾸면 `tests/test_planner.py`부터 고치세요.
- 작업 기록 코드는 `tablely/worklog.py`에 있습니다. `tools/agent_log.py`는 hooks가 설치 없이 부르기 위한 얇은 스크립트입니다.
- Tablely는 CLI 도구입니다. 웹 UI는 두지 않습니다. 달성률과 재개 요약은 `tablely/progress.py`에 있고, 터미널 출력(`status`의 막대 등)은 `tablely/board_view.py`, 재개 고르기는 `cli.py`의 `_cmd_resume`에 있습니다.
- 여러 에이전트 조율은 `tablely/ledger.py`(공용 장부)와 `tablely/runner.py`의 `_tick`이 담당합니다. CPU↔GPU 전환 요청은 `runner.py`의 `_switch_wishes`가 정합니다.
- `tablely/stream.py`(순차 업로드)의 CUDA 경로(`CudaBackend`, 다중 GPU `map_chunks`)는 GPU 없는 환경에서는 실행되지 않습니다. 파이프라인 로직은 `tests/test_stream.py`의 가짜 backend로 검증합니다. GPU 서버에서 작업하게 되면 이 경로부터 실제로 돌려 보세요.
- 테스트는 `tests/conftest.py`가 `TABLELY_HOME`을 임시 디렉터리로 바꾸므로 실제 `~/.tablely`를 건드리지 않습니다.
