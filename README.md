# Tablely

한 대의 머신에서 여러 학습 작업을 동시에 돌릴 때, **GPU와 CPU 코어를 중요도(priority) 순으로 나눠주는** 작은 스케줄러입니다.

한 서버에 학습 여러 개를 그냥 띄우면 이런 일이 생깁니다.

- 여러 작업이 같은 GPU에 올라가 OOM이 나거나, 반대로 어떤 GPU는 놀고 있음
- 작업마다 PyTorch/NumPy가 "코어 수만큼" 스레드를 띄워 CPU가 과다 구독 → 전부 느려짐
- GPU 작업이 데이터 로딩에 쓸 CPU를 못 받아서 GPU 사용률이 떨어짐
- 중요한 실험이 덜 중요한 실험 뒤에서 기다림

Tablely는 작업 목록을 받아서:

1. **GPU**: 중요도가 높은 작업부터 GPU를 통째로 배정합니다 (`CUDA_VISIBLE_DEVICES`).
2. **CPU 대체 실행**: GPU가 모자라면 `device = "any"` 작업은 CPU에서 돌립니다. GPU와 CPU 둘 다 놀지 않게 합니다.
3. **CPU 코어**: 작업마다 겹치지 않는 코어 집합을 주고 그 코어에 고정(CPU affinity)합니다. 스레드 수 환경변수(`OMP_NUM_THREADS` 등)도 맞춰줍니다.
4. **중요도 비례 분배**: 최소 코어를 보장한 뒤 남는 코어는 중요도에 비례해 나눕니다.
5. **재분배**: 작업이 끝날 때마다 다시 계획합니다. 대기 중인 작업을 시작하고, 실행 중인 작업의 코어를 늘리거나 줄입니다.
6. **여러 에이전트**: 같은 머신에서 여러 에이전트(AI 에이전트나 사람)가 각자 `tablely run`을 해도 GPU와 코어를 함께 계획합니다. 누가 무엇을 왜 돌리고 있는지는 자동으로 기록합니다 (`tablely status`, `tablely history`).

## 설치

```bash
pip install -e .            # YAML 작업 파일도 쓰려면: pip install -e '.[yaml]'
```

Python 3.10 이상이 필요하고 Linux를 권장합니다. 코어 고정은 Linux에서만 되고, macOS에서는 스레드 수 환경변수만 적용됩니다.

## 빠른 시작

```bash
tablely resources                                     # Tablely가 쓸 CPU/GPU 확인
tablely plan examples/jobs.toml                       # 배치 미리보기 (아무것도 실행 안 함)
tablely plan examples/jobs.toml --cpus 16 --gpus 2    # 다른 사양의 서버를 가정해서 미리보기
tablely run  examples/jobs.toml                       # 실행
```

`tablely plan examples/jobs.toml --cpus 16 --gpus 2` 결과 (코어 0은 `reserve_cpus = 1`로 OS용으로 남김):

```
resources: 15 CPU core(s) [1-15], 2 GPU(s) [0,1]
policy: strict priority

  JOB            PRIO  WANTS   PLACED ON  CORES      STATUS
  llm-finetune   10    gpu x1  GPU 0      1-4 (4)    start
  vision-cnn     5     any x1  GPU 1      5-8 (4)    start
  xgboost-sweep  3     cpu     CPU        9-13 (5)   start
  tabular-mlp    1     any x1  CPU        14-15 (2)  start
```

GPU 2장은 중요도 10, 5인 작업이 가져가고, GPU를 못 받은 `tabular-mlp`(any)는 CPU에서 돕니다. CPU 전용 `xgboost-sweep`(중요도 3)은 `tabular-mlp`(중요도 1)보다 남는 코어를 더 받습니다.

실행 로그 예시 (예시 파일을 4코어 머신에 맞게 줄이고 `--gpus 2`로 실행):

```
[01:45:39] start  llm-finetune   prio 10  GPU 0  cores 0-1 (2)
[01:45:39] start  vision-cnn     prio 5  GPU 1  cores 2-3 (2)
[01:45:39] wait   xgboost-sweep  needs 2 core(s), 0 free
[01:45:39] wait   tabular-mlp    cores held for a higher-priority job
[01:45:42] done   vision-cnn     ok in 4s
[01:45:42] start  xgboost-sweep  prio 3  CPU  cores 2-3 (2)
[01:45:42] wait   tabular-mlp    needs 1 core(s), 0 free
[01:45:45] done   llm-finetune   ok in 7s
[01:45:45] resize xgboost-sweep  cores 2-3 (2) -> 0,2-3 (3)
[01:45:45] start  tabular-mlp    prio 1  GPU 0  cores 1 (1)
...
```

## 배분 규칙

### 1. GPU: 중요도 순, 통째로

대기 중인 작업을 중요도 높은 순서대로 봅니다. 중요도가 같으면 파일에 적힌 순서를 따릅니다. GPU가 필요한 작업은 비어 있는 GPU를 `gpus`개만큼 통째로 받습니다. 한 GPU를 두 작업이 나눠 쓰는 일은 없습니다. 이미 실행 중인 작업의 GPU는 절대 옮기지 않습니다.

### 2. `device`: 어디서 돌릴 수 있는가

| 값 | 의미 |
|---|---|
| `gpu` (기본) | GPU에서만 돕니다. GPU가 없으면 기다립니다. |
| `cpu` | CPU에서만 돕니다. GPU는 보이지 않게 합니다 (`CUDA_VISIBLE_DEVICES=""`). |
| `any` | GPU가 비어 있으면 GPU, 없으면 CPU에서 바로 시작합니다. 스크립트가 `{device}` 또는 `TABLELY_DEVICE`를 보고 장치를 골라야 합니다. |

### 3. CPU 코어: 최소 보장 + 중요도 비례

- 모든 작업은 `cpus`개(최소 코어)를 보장받습니다.
- 남는 코어는 **중요도에 비례해서** 나눕니다 (가중 max-min 공정 분배). 중요도 4인 작업은 중요도 2인 작업의 약 2배를 받습니다.
- `max_cpus`가 있으면 그 이상은 받지 않습니다.
- **GPU에서 도는 작업**은 `max_cpus`를 따로 적지 않으면 `cpus`개에서 늘어나지 않습니다. GPU 학습에서 CPU는 주로 데이터 로딩에 쓰이니 남는 코어는 CPU 작업에 돌리는 게 낫기 때문입니다.
- 작업마다 코어 집합이 겹치지 않습니다. 아무도 못 받는 코어(모두 `max_cpus`에 걸린 경우)는 비워 둡니다.

예: 코어 12개에 CPU 작업 A(중요도 2), B(중요도 1) → A 8개, B 4개.

### 4. 엄격한 우선순위 (기본) vs `backfill`

기본값은 **엄격한 우선순위**입니다. 중요한 작업이 자원이 모자라 기다리는 동안에는 덜 중요한 작업이 그 자원을 가져갈 수 없습니다.

- GPU를 기다리는 작업이 있으면 남는 GPU는 그 작업 몫으로 남겨 둡니다. 그 사이 덜 중요한 `any` 작업은 CPU에서 돕니다.
- 기다리는 작업의 최소 코어도 미리 빼 둡니다.

그래서 중요한 작업이 계속 밀리는(starvation) 일이 없습니다. 대신 자원이 잠깐 놀 수 있습니다.
`backfill = true`(또는 `--backfill`)로 바꾸면 덜 중요한 작업도 노는 자원을 바로 씁니다. 활용률은 올라가지만, 큰 작업(예: GPU 2장)은 더 오래 기다릴 수 있습니다.

### 5. 실행 중 재분배

작업이 끝나면 다시 계획합니다.

- 기다리던 작업을 시작합니다.
- 실행 중인 작업들의 코어 수를 위 규칙대로 다시 계산해서 늘리거나 줄입니다.
  - 기존 코어는 가능한 한 유지합니다.
  - 작업의 모든 스레드와 자식 프로세스(DataLoader 워커 등)에 새 affinity를 적용합니다.
- 줄일 때도 최소 코어(`cpus`) 밑으로는 내려가지 않습니다.

## 작업 파일

TOML, YAML, JSON을 지원합니다. 전체 예시는 [`examples/jobs.toml`](examples/jobs.toml)에 있습니다.

```toml
log_dir = "logs"          # 작업별 로그 <name>.log (작업 파일 기준 상대경로)
backfill = false
task = "resnet lr sweep"  # 이 실행의 목적. 다른 에이전트가 status/history에서 봄 (--task로 덮어쓰기)

[resources]               # 모두 생략 가능
cpus = "0-15"             # 정수 = 앞에서부터 N개, 문자열/리스트 = 그 코어들. 기본: 사용 가능한 전체
gpus = [0, 1]             # 정수 = N장, 리스트/문자열 = 그 ID들. 기본: CUDA_VISIBLE_DEVICES → nvidia-smi
reserve_cpus = 1          # 앞쪽 N개 코어는 OS/Tablely용으로 남김

[[jobs]]
name = "llm-finetune"     # 영문/숫자/._- (로그 파일 이름으로 씀)
command = "python train.py --device {device} --workers {cpus}"
priority = 10             # 양수. 높을수록 먼저 + 남는 코어도 더 받음 (기본 1)
device = "gpu"            # gpu | cpu | any (기본 gpu)
gpus = 1                  # GPU에서 돌 때 필요한 장수 (기본 1)
cpus = 4                  # 최소 보장 코어 (기본 1)
max_cpus = 8              # 코어 상한 (기본: GPU면 cpus, CPU면 무제한)
env = { WANDB_PROJECT = "exp" }
cwd = "."                 # 작업 디렉터리 (기본: 작업 파일이 있는 곳)
shell = false             # true면 command를 셸로 실행 (파이프, && 등)
task = "lr 1e-3, warmup 500"  # 이 작업만의 목적 (기본: 위의 task)
```

`command`는 문자열 또는 리스트입니다. 아래 자리표시자는 작업 시작 시점의 값으로 바뀝니다.

| 자리표시자 | 값 |
|---|---|
| `{device}` | `cuda` 또는 `cpu` |
| `{gpus}` | 배정된 GPU ID (`0,1`), CPU면 빈 문자열 |
| `{num_gpus}` | GPU 장수 |
| `{cpus}` | 시작 시점 코어 수 |
| `{cpu_list}` | 코어 목록 (`4-7`) |
| `{name}` | 작업 이름 |

CLI 옵션 `--cpus`, `--gpus`, `--reserve-cpus`, `--backfill`, `--log-dir`, `--task`는 작업 파일 설정보다 우선합니다.

## 학습 스크립트와 연동

Tablely는 작업마다 다음 환경변수를 넣어 줍니다.

| 변수 | 내용 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | 배정된 GPU (CPU 작업은 `""` → GPU 안 보임) |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` (nvidia-smi 번호와 일치시키기 위함, 이미 설정돼 있으면 유지) |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` | 시작 시점 코어 수 (작업의 `env`에 직접 적으면 그 값을 존중) |
| `TABLELY_DEVICE` | `cuda` 또는 `cpu` |
| `TABLELY_GPUS`, `TABLELY_NUM_CPUS`, `TABLELY_CPU_LIST` | 배정 내역 |
| `TABLELY_JOB`, `TABLELY_PRIORITY` | 작업 이름, 중요도 |
| `TABLELY_AGENT`, `TABLELY_TASK` | 이 작업을 띄운 에이전트, 작업 목적 |
| `TABLELY_HOME`, `TABLELY_RUN`, `TABLELY_JOB_KEY` | 공용 장부 위치와 이 작업의 키 (`client.progress`가 사용) |

환경변수만 읽어도 되지만, 선택적으로 쓸 수 있는 헬퍼도 있습니다.

```python
from tablely import client

device = client.device()        # "cuda" / "cpu" (Tablely 밖에서 실행하면 기본값 "cpu")
model.to(device)

for epoch in range(epochs):
    client.sync_torch_threads()  # 코어가 늘거나 줄었으면 torch 스레드 수를 맞춤
    ...
    client.progress(f"epoch {epoch + 1}/{epochs}, val acc {acc:.3f}")  # tablely status에 표시
```

`any` 작업을 쓸 때는 장치에 따라 배치 크기 등을 바꾸고 싶을 수 있습니다. 그럴 때는 `command`에 `--device {device}`를 넘겨서 스크립트에서 분기하면 됩니다.

## 여러 에이전트가 한 머신을 같이 쓸 때

AI 에이전트 여러 개(또는 사람)가 같은 머신에서 각자 `tablely run`을 실행해도 됩니다. 모든 `tablely run`은 머신 공용 장부(`~/.tablely`, `TABLELY_HOME` 또는 `--home`으로 변경)에 자동으로 등록됩니다. 그래서 따로 설정하지 않아도 다음이 됩니다.

- **자원을 같이 계획합니다.** 다른 에이전트가 쓰고 있는 GPU와 코어는 건드리지 않습니다. 중요도는 에이전트 구분 없이 비교하고, 코어 재분배도 머신 전체 기준으로 합니다.
- **누가 무엇을 하는지 자동으로 기록합니다.** 에이전트 이름, 목적(task), 배정된 GPU와 코어, 시작·종료·실패·코어 변경, 코드 버전(git 커밋, 브랜치, 수정 여부), 학습 진행 상황을 남깁니다.

```bash
# 에이전트마다 이름과 목적을 붙여서 실행 (이름은 TABLELY_AGENT 환경변수로도 지정 가능)
tablely run sweep.toml --agent claude-a --task "resnet lr sweep"
tablely run vit.toml   --agent claude-b --task "vit augmentation study"

tablely status                 # 지금 누가 무엇을 어디서 돌리는지 (--json 가능)
tablely history                # 지난 기록 (--agent, --job, -n, --json)
tablely note --agent claude-a "lr sweep 끝나면 baseline과 비교 예정"   # 지금 하는 일 메모
tablely plan vit.toml          # 다른 에이전트의 작업을 고려한 배치 미리보기
```

`tablely status` 예시 (GPU 1장 머신에서 두 에이전트가 동시에 실행):

```
machine: 4 CPU core(s) [0-3], 1 GPU(s) [0]   policy: strict priority

agents:
  AGENT     RUNS  RUNNING  WAITING  TASK                    NOTE
  claude-a  1     1        0        resnet lr sweep         -
  claude-b  1     1        1        vit augmentation study  -

jobs:
  AGENT     JOB           TASK                    PRIO  STATE    ON     CORES    TIME  INFO
  claude-a  lr-1e-3       resnet lr sweep         5     running  GPU 0  0 (1)    2s    epoch 3/3 (0s ago)
  claude-b  feature-prep  precompute features     2     running  CPU    1-3 (3)  2s    epoch 3/3 (0s ago)
  claude-b  vit-aug       vit augmentation study  9     waiting  -      -        2s    needs 1 GPU(s), 0 free
```

`tablely history` 예시:

```
TIME                 AGENT     EVENT      JOB           DETAIL                        TASK
2026-10-08 05:32:00  claude-a  start      lr-1e-3       prio 5 GPU 0 cores 0 (1)      resnet lr sweep
2026-10-08 05:32:00  claude-b  start      feature-prep  prio 2 CPU cores 1-3 (3)      precompute features
2026-10-08 05:32:00  claude-b  wait       vit-aug       needs 1 GPU(s), 0 free        vit augmentation study
2026-10-08 05:32:02  claude-a  note       -             lr sweep 끝나면 baseline과 비교 예정
2026-10-08 05:32:03  claude-a  done       lr-1e-3       ok in 4s                      resnet lr sweep
2026-10-08 05:32:03  claude-b  start      vit-aug       prio 9 GPU 0 cores 0 (1)      vit augmentation study
```

세부 동작:

- 처음 실행된 `tablely run`이 머신의 자원 풀(코어, GPU, 정책)을 정합니다. 누군가 실행 중인 동안에는 나중에 온 에이전트도 그 풀을 따릅니다.
- 더 중요한 작업이 와도 다른 에이전트가 이미 돌리고 있는 작업은 멈추지 않습니다 (선점 없음). 위 예시처럼 기다렸다가 자원이 비면 시작합니다.
- `tablely` 프로세스가 비정상 종료되면(`kill -9` 등), 그 실행의 대기 작업은 장부에서 지워집니다. 이미 돌고 있던 작업은 `orphan`으로 표시되고, 끝날 때까지 자원을 계속 차지합니다. 그 GPU에 다른 작업이 올라가지 않게 하기 위해서입니다.
- 학습 스크립트에서 `client.progress(...)`를 부르면 status에 진행 상황이 나옵니다. 공용 파일을 잠그고 쓰므로 스텝마다가 아니라 에폭마다 정도로 부르세요.
- 장부 디렉터리에는 `state.json`(현재 상태)과 `history.jsonl`(이벤트, 한 줄에 JSON 하나)이 있습니다. 파일 잠금(`flock`)으로 동시 접근을 막습니다.
- 같은 머신의 같은 사용자끼리만 공유됩니다. 다른 머신과는 공유되지 않습니다.

## 그 밖의 동작

- 각 작업은 자기만의 프로세스 그룹에서 실행됩니다. 메인 프로세스가 끝나면 남아 있는 자식 프로세스(DataLoader 워커 등)도 정리해서 GPU 메모리와 코어를 확실히 돌려받습니다.
- `Ctrl+C`나 `SIGTERM`을 받으면 모든 작업에 SIGTERM을 보내고, 10초 뒤에도 살아 있으면 SIGKILL을 보냅니다.
- 한 작업이 실패해도 나머지는 계속 돕니다. 마지막에 요약표를 출력합니다.
- 종료 코드: `0` 모두 성공, `1` 실패한 작업 있음, `2` 설정 오류, `130` 중단됨.
- 어떤 자원 상태에서도 실행될 수 없는 작업(예: GPU 2장 머신에서 `gpus = 3`)은 시작 전에 오류로 알려줍니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `tablely/spec.py` | `JobSpec`: 작업 정의와 검증 |
| `tablely/resources.py` | CPU/GPU 감지, `Inventory` |
| `tablely/planner.py` | 배분 로직 (순수 함수, I/O 없음 → 테스트/미리보기 용이) |
| `tablely/runner.py` | 프로세스 실행, 감시, 재분배, 종료 처리 (공용 장부를 통해 다른 에이전트와 함께 계획) |
| `tablely/ledger.py` | 머신 공용 장부: 실행/작업 등록, 죽은 실행 정리, 이벤트 기록 |
| `tablely/board_view.py` | `tablely status` / `history` 출력 |
| `tablely/affinity.py` | 프로세스 그룹 전체 코어 고정 (Linux) |
| `tablely/config.py` | TOML/YAML/JSON 작업 파일 로딩 |
| `tablely/cli.py` | `tablely resources / plan / run / status / history / note` |
| `tablely/client.py` | 학습 스크립트용 선택적 헬퍼 (`device`, `progress`, `note` ...) |
| `tools/agent_log.py` | 이 저장소를 개발하는 코딩 에이전트의 작업 기록 (아래 참고) |

테스트: `pip install -e '.[test,yaml]' && pytest`

## 이 저장소를 여러 코딩 에이전트로 개발할 때

Claude Code 같은 코딩 에이전트 여러 개가 이 저장소를 동시에 작업하면, 각 에이전트가 지금 무엇을 하는지 자동으로 기록합니다. 설정은 `.claude/settings.json`의 hooks에 들어 있어서 저장소를 열면 바로 동작합니다.

- 세션마다 `.agents/sessions/<세션ID 앞 8자리>.json` 파일 하나에 다음을 남깁니다.
  - 브랜치
  - 상태 (`working`: 작업 중, `idle`: 응답 끝, `ended`: 세션 종료)
  - 작업 내용
  - 최근 프롬프트 첫 줄
  - 수정한 파일
- 파일이 세션별로 나뉘어 있어서 여러 에이전트가 동시에 써도 충돌하지 않습니다. 작업과 함께 커밋하면 다른 머신의 에이전트도 그 브랜치에서 볼 수 있습니다.
- 새 세션이 시작되면 다른 에이전트들이 지금 하는 일을 자동으로 보여줍니다. 대상은 이 체크아웃, 같은 clone의 다른 worktree, 모든 원격 브랜치입니다.
- 작업 내용은 에이전트가 직접 적은 것이 우선입니다: `python3 tools/agent_log.py task "NUMA 인지 코어 배정"`. 적은 게 없으면 최근 프롬프트 첫 줄을 씁니다.
- 전체 현황: `python3 tools/agent_log.py board` (`--fetch`를 붙이면 원격을 먼저 가져옴, `--all`이면 오래전에 끝난 세션도 표시)
- 이 저장소는 공개 저장소이므로 프롬프트 첫 줄(최대 120자)도 커밋되면 공개됩니다. 남기기 싫으면 `.claude/settings.json`의 `"env"`에 `"AGENT_LOG_PROMPTS": "0"`을 넣으세요. 그러면 에이전트가 `task`로 적은 내용만 남습니다.

## 한계와 다음 단계

지금은 에이전트마다 `tablely run`이 자기 작업 목록을 끝까지 돌리고, 머신 공용 장부로 서로 조율하는 구조입니다. 서브시스템으로 키우려면 이런 것들이 다음 후보입니다.

- **데몬 + `tablely submit`**: 이미 돌고 있는 실행에 작업을 추가하거나 취소. 지금은 새 작업 목록마다 `tablely run`을 하나 더 띄워야 합니다.
- **선점(preemption)**: 더 중요한 작업이 나중에 들어왔을 때 덜 중요한 작업을 체크포인트 후 멈추고 나중에 재개. 지금은 실행 중인 작업을 멈추지 않습니다.
- **CPU → GPU 이동**: CPU에서 시작한 `any` 작업은 나중에 GPU가 비어도 옮겨가지 않습니다. 선점과 같은 체크포인트/재시작 방식이 필요합니다.
- **GPU 나눠 쓰기**: 작은 작업 여러 개가 한 GPU를 공유 (MPS, MIG, 메모리 비율 제한).
- **토폴로지 인지**: GPU와 같은 NUMA 노드의 코어를 우선 배정.
- **사용률 기반 조정**: 실제 GPU/CPU 사용률을 보고 코어를 재분배. 예를 들어 GPU 사용률이 낮으면 데이터 로딩 코어를 늘립니다.
- **스레드 수 동적 조정**: 지금은 코어가 바뀌어도 `OMP_NUM_THREADS`는 시작 시점 값 그대로라서, 스크립트가 `client.sync_torch_threads()`를 불러야 합니다.
- AMD ROCm GPU 자동 감지 (지금은 `--gpus`로 직접 지정해야 함).
