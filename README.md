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
6. **다중 GPU와 장치 전환**: GPU 여러 장을 한 작업에 줄 수 있고, 아무도 기다리지 않으면 남는 GPU를 더 줍니다 (`max_gpus`). 실행 중인 작업도 GPU가 비면 GPU로, 더 중요한 작업이 GPU를 기다리면 CPU로 옮깁니다 (체크포인트 후 재시작, `switchable`).
7. **큰 데이터 순차 업로드**: GPU 메모리나 RAM보다 큰 데이터를 조각으로 나눠 디스크 → RAM → PCIe 버스 → GPU 순서로 계속 흘려 보냅니다. CPU에서는 L3 캐시에 맞는 크기로 쪼개서 처리합니다 (`tablely.stream`).
8. **여러 에이전트와 핸드오프**: 같은 머신에서 여러 에이전트(AI 에이전트나 사람)가 각자 `tablely run`을 해도 GPU와 코어를 함께 계획합니다. 누가 무엇을 왜 돌리고 있는지는 자동으로 기록합니다. 그래서 다음 에이전트는 따로 인수인계 문서를 받지 않아도 `tablely brief`로 이어받을 수 있습니다.

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

### 6. 다중 GPU와 자동 분배: `max_gpus`

- `gpus = 2`이면 그 작업은 항상 GPU 2장을 통째로 받습니다.
- `max_gpus = 4`(또는 `"all"`)를 같이 적으면, 시작할 때 GPU를 기다리는 작업이 없는 경우 최대 4장까지 더 받습니다. 남는 GPU가 여러 작업에 걸리면 중요도 비례로 나눕니다.
- 누군가 GPU를 기다리고 있으면 `gpus`만큼만 받습니다.
- 실행 중인 작업의 GPU 수는 바뀌지 않습니다. 이미 CUDA를 초기화한 프로세스의 GPU를 늘릴 수 없기 때문입니다.
- 몇 장을 받았는지는 `{num_gpus}`로 알 수 있습니다:
  `command = "torchrun --nproc_per_node {num_gpus} train.py"`
- 한 프로세스가 받은 GPU 여러 장에 데이터를 나눠 처리하려면 `tablely.stream.map_chunks`를 쓰세요 ([아래](#큰-데이터-순차-업로드-tablelystream)).

## 작업 파일

TOML, YAML, JSON을 지원합니다. 전체 예시는 [`examples/jobs.toml`](examples/jobs.toml)에 있습니다.

```toml
log_dir = "logs"          # 작업별 로그 <name>.log (작업 파일 기준 상대경로)
backfill = false
task = "resnet lr sweep"  # 이 실행의 목적. 다른 에이전트가 status/history에서 봄 (--task로 덮어쓰기)
switch_grace = 60         # switchable 작업이 장치 이동 요청을 받기 전 최소 실행 시간(초)
max_switches = 5          # 작업당 장치 이동 횟수 상한

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
max_gpus = 4              # 남는 GPU가 있으면 시작할 때 이만큼까지 더 받음 ("all" = 전부)
cpus = 4                  # 최소 보장 코어 (기본 1)
max_cpus = 8              # 코어 상한 (기본: GPU면 cpus, CPU면 무제한)
env = { WANDB_PROJECT = "exp" }
cwd = "."                 # 작업 디렉터리 (기본: 작업 파일이 있는 곳)
shell = false             # true면 command를 셸로 실행 (파이프, && 등)
task = "lr 1e-3, warmup 500"  # 이 작업만의 목적 (기본: 위의 task)
switchable = false        # true면 체크포인트 후 CPU<->GPU로 옮겨질 수 있음 (아래 참고)
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
| `TABLELY_SWITCHABLE`, `TABLELY_RESTARTS` | 장치 이동 가능 여부, 지금까지 이동한 횟수 (재시작이면 1 이상) |
| `TABLELY_CONTROL`, `TABLELY_REPLY` | 장치 이동 요청을 주고받는 파일 (`client.switch_requested` / `exit_for_switch`가 사용) |

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

## 실행 중 CPU ↔ GPU 전환: `switchable`

이미 시작한 프로세스의 장치를 그 자리에서 바꿀 수는 없습니다. 그래서 작업과 Tablely가 협력합니다. 작업 파일에 `switchable = true`를 적은 `device = "any"` 작업에 적용됩니다.

1. Tablely가 작업에게 이동을 요청합니다.
   - **GPU로**: GPU가 바빠서 CPU에서 시작했는데 GPU가 비었고, 그 GPU를 기다리는 작업이 없을 때
   - **CPU로**: 더 중요한 GPU 전용 작업이 GPU를 기다리는데, 이 작업이 GPU를 내주면 그 작업이 시작할 수 있을 때. 중요도가 낮은 작업부터 요청합니다.
2. 작업은 적당한 시점(에폭 끝 등)에 요청을 확인하고, 체크포인트를 저장한 뒤 종료 코드 75로 끝납니다.
3. Tablely가 작업을 다시 대기열에 넣고 새 장치에서 재시작합니다. 재시작된 작업은 `client.restarts()`가 1 이상입니다.

```python
from tablely import client

start = load_checkpoint() if client.restarts() else 0
for epoch in range(start, epochs):
    train_one_epoch()
    if client.switch_requested():         # "gpu" 또는 "cpu"
        save_checkpoint(epoch + 1)
        client.exit_for_switch()          # Tablely가 다른 장치에서 다시 시작

# GPU 메모리가 부족하면 스스로 CPU로 옮겨갈 수도 있음 (이후 계속 CPU)
try:
    step()
except torch.cuda.OutOfMemoryError:
    save_checkpoint(epoch)
    client.exit_for_switch("cpu")
```

- 요청을 무시하면 아무 일도 일어나지 않습니다. 이동 이유가 사라지면 요청도 철회됩니다. 예를 들어 비었던 GPU를 다른 작업이 먼저 가져가면 그렇습니다.
- 너무 자주 오가지 않도록, 작업이 `switch_grace`초(기본 60) 이상 돈 뒤에만 요청합니다. 작업당 `max_switches`번(기본 5)을 넘으면 더 요청하지 않습니다.
- 여러 에이전트가 있어도 모두 같은 공용 장부를 보고 같은 결론을 내립니다. 각자 자기 작업에만 요청하므로, 빈 GPU 하나를 두 작업이 동시에 노리는 일이 없습니다.
- `switchable`이 아닌 작업이 75로 끝나면 그냥 실패로 처리합니다.
- 예시: [`examples/stream_train.py`](examples/stream_train.py). 아래 실행 로그는 CPU에서 시작해 epoch 101에 체크포인트하고, GPU가 비자 GPU로 옮겨 이어서 끝낸 기록입니다.

```
[07:26:38] start  hog           prio 9  GPU 0  cores 0 (1)
[07:26:38] start  stream-train  prio 1  CPU  cores 1-3 (3)
[07:26:40] done   hog           ok in 2s
[07:26:40] resize stream-train  cores 1-3 (3) -> 0-3 (4)
[07:26:40] switch stream-train  asked to checkpoint and move to GPU: 1 GPU(s) free and nobody waiting for them
[07:26:40] requeue stream-train  checkpointed on CPU; restarting on GPU (switch 1/5)
[07:26:40] start  stream-train  prio 1  GPU 0  cores 0 (1)
[07:26:50] done   stream-train  ok in 10s
```

## 큰 데이터 순차 업로드: `tablely.stream`

GPU 메모리(또는 RAM)보다 큰 데이터를 한 번에 올리지 않고, 순서대로 조각(chunk)을 흘려 보냅니다. 학습 스크립트 안에서 씁니다. CPU 경로는 NumPy만 있으면 되고, GPU 경로는 PyTorch(CUDA)가 필요합니다. 설치: `pip install 'tablely[stream]'`

```python
from tablely import stream

data = stream.open_array("features.npy")   # 메모리 맵: 아직 아무것도 읽지 않음 (RAM보다 커도 됨)
data = stream.preload(data)                 # 남은 RAM의 절반 안에 들어가면 RAM에 올림, 아니면 디스크에서 스트리밍

for x in stream.chunks(data):               # Tablely가 준 장치(GPU 또는 CPU)로 한 조각씩
    loss = model(x)

# 받은 GPU 전부에 조각을 나눠서 동시에 처리 (GPU가 없으면 CPU 코어 여러 개로)
outputs = stream.map_chunks(lambda x: model(x).cpu(), data)
```

조각이 GPU까지 가는 길:

```
디스크/RAM --(읽기 스레드)--> 고정(pinned) RAM 버퍼 --(PCIe DMA, 전용 CUDA 스트림)--> GPU
```

- **겹쳐서 진행합니다.** 모델이 i번째 조각을 계산하는 동안 i+1번째는 버스를 건너고 있고, i+2번째는 디스크에서 RAM으로 읽히고 있습니다. 그래서 디스크나 버스가 계산보다 실제로 느릴 때만 GPU가 기다립니다.
- **고정 RAM 버퍼 몇 개(기본 3개)를 돌려 씁니다.** 페이지가 고정된 메모리라 GPU의 복사 엔진이 CPU를 거치지 않고 직접 가져갑니다. DMA가 아직 읽고 있는 버퍼에는 덮어쓰지 않습니다.
- **조각 크기**:
  - GPU: 기본 64 MiB. 남은 GPU 메모리와 RAM 예산(사용 가능한 RAM의 1/4)에 맞춰 줄입니다.
  - CPU: L3 캐시의 절반. 한 조각을 처리하는 동안 데이터가 캐시에 머물러 있게 하기 위해서입니다. `map_chunks`로 여러 스레드가 같은 L3를 나눠 쓰면 스레드 수만큼 더 작게 나눕니다.
- **RAM보다 큰 파일**은 메모리 맵으로 열어 순서대로 읽습니다. RAM에는 버퍼 몇 개 분량만 올라갑니다.
- **장치는 Tablely를 따릅니다.**
  - GPU 작업이면 받은 GPU들(`CUDA_VISIBLE_DEVICES`)에 나눕니다.
  - CPU에 배치됐거나 PyTorch가 CUDA를 못 쓰면 CPU로 계산합니다.
- `chunks(...)`의 `.stats`로 실제 상황을 볼 수 있습니다: 옮긴 바이트, 걸린 시간, 데이터를 기다린 시간. 기다린 시간이 길면 디스크나 버스가 병목입니다.
- **L3 캐시**는 프로그램이 "여기에 올려라"라고 지정할 수 없고, GPU로 가는 DMA는 RAM에서 읽습니다. 그래서 GPU 경로는 RAM(고정 버퍼)을 거치고, L3는 CPU 계산을 캐시 크기에 맞춰 쪼개는 데 씁니다.

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
tablely note --agent claude-a "lr sweep 끝나면 baseline과 비교 예정"   # 지금 하는 일, 다음 할 일 메모
tablely brief                  # 다음 에이전트에게 넘길 핸드오프 요약 (--hours, --agent, --json)
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

### 핸드오프: `tablely brief`

에이전트가 바뀌거나 세션이 끝나도 따로 인수인계 문서를 쓸 필요가 없습니다. 새로 온 에이전트는 `tablely brief` 출력을 그대로 컨텍스트에 넣으면 됩니다.

```
# Tablely brief (2026-10-08 07:10, last 24h)
machine: 4 core(s), GPUs [0], strict priority

## Now
- [claude-b] vit-mixup (vit augmentation study): running on GPU 0, cores 0 (1), 3s so far; progress: epoch 1/3

## Finished (newest first)
- 07:10 [claude-a] lr-1e-3 (resnet lr sweep): ok in 4s; last progress: epoch 3/3

## Notes from agents (newest first)
- 07:10 [claude-a] lr 1e-3 is best so far; next: try 3e-4 with warmup
```

- **Now**: 지금 돌고 있거나 기다리는 작업과 그 진행 상황입니다.
- **Finished**: 끝난 작업의 결과입니다. 마지막으로 보고한 진행 상황(지표), 코드 커밋, 실패했다면 로그 위치가 함께 나옵니다.
- **Notes**: 에이전트가 `tablely note`로 남긴 "다음에 할 일"입니다.
- `--agent claude-a`로 한 에이전트의 작업만, `--hours 6`으로 최근 6시간만 볼 수 있습니다.

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
| `tablely/cli.py` | `tablely resources / plan / run / status / history / note / brief` |
| `tablely/client.py` | 학습 스크립트용 선택적 헬퍼 (`device`, `progress`, `note`, `switch_requested`, `exit_for_switch` ...) |
| `tablely/stream.py` | 큰 데이터 순차 업로드: 메모리 맵 → 고정 RAM 버퍼 → GPU, CPU는 L3 크기 블록 |
| `tools/agent_log.py` | 이 저장소를 개발하는 코딩 에이전트의 작업 기록 (아래 참고) |

테스트: `pip install -e '.[test,yaml]' && pytest` (GPU 경로의 CUDA 복사 부분은 GPU 없는 환경에서는 실행되지 않습니다)

## 이 저장소를 여러 코딩 에이전트로 개발할 때

Claude Code 같은 코딩 에이전트 여러 개가 이 저장소를 동시에 작업하면, 각 에이전트가 지금 무엇을 하는지 자동으로 기록합니다. 이 기록이 세션 사이의 핸드오프 역할을 합니다. 설정은 `.claude/settings.json`의 hooks에 들어 있어서 저장소를 열면 바로 동작합니다.

- 세션마다 `.agents/sessions/<세션ID 앞 8자리>.json` 파일 하나에 다음을 남깁니다.
  - 브랜치
  - 상태 (`working`: 작업 중, `idle`: 응답 끝, `ended`: 세션 종료)
  - 작업 내용
  - 최근 프롬프트 첫 줄
  - 수정한 파일
- 파일이 세션별로 나뉘어 있어서 여러 에이전트가 동시에 써도 충돌하지 않습니다. 작업과 함께 커밋하면 다른 머신의 에이전트도 그 브랜치에서 볼 수 있습니다.
- 커밋되는 이 파일은 에이전트가 파일을 수정할 때만 갱신됩니다. 그래서 로그 때문에 작업 트리가 혼자 더러워지는 일은 없습니다. 상태와 프롬프트 같은 실시간 정보는 커밋되지 않는 `.git/agent-sessions/`에 바로 반영되고, 같은 clone의 모든 worktree가 이 디렉터리를 함께 봅니다.
- **새 세션은 첫 프롬프트 전에 핸드오프를 자동으로 받습니다.** 대상은 이 체크아웃, 같은 clone의 다른 worktree, 모든 원격 브랜치입니다. 받는 내용은 다음과 같습니다.
  - 최근 세션 3개가 무엇을 하고 있었고 어디서 멈췄는지
  - 어떤 파일을 고쳤는지
  - 지금 누가 작업 중인지
- 어디서 멈췄는지는 두 가지 방법으로 남습니다.
  - **자동**: 에이전트가 응답을 마칠 때 마지막 답변(최대 600자)이 남습니다.
  - **직접**: 작업을 마치기 전에 `python3 tools/agent_log.py handoff "끝낸 것 / 다음 할 것 / 주의할 점"`으로 적습니다. 직접 적은 핸드오프는 바로 커밋할 사본에 들어갑니다. 그래서 작업과 함께 푸시하면 다른 머신의 새 세션도 받습니다.
- 새 세션이 받는 내용을 미리 보려면: `python3 tools/agent_log.py brief`
- 작업 내용은 에이전트가 직접 적은 것이 우선입니다: `python3 tools/agent_log.py task "NUMA 인지 코어 배정"`. 적은 게 없으면 최근 프롬프트 첫 줄을 씁니다.
- 전체 현황: `python3 tools/agent_log.py board` (`--fetch`를 붙이면 원격을 먼저 가져옴, `--all`이면 오래전에 끝난 세션도 표시)
- 이 저장소는 공개 저장소이므로 프롬프트 첫 줄(최대 120자)과 에이전트의 마지막 답변도 커밋되면 공개됩니다. 남기기 싫으면 `.claude/settings.json`의 `"env"`에 `"AGENT_LOG_PROMPTS": "0"`을 넣으세요. 그러면 에이전트가 `task`와 `handoff`로 직접 적은 내용만 남습니다.

## 한계와 다음 단계

지금은 에이전트마다 `tablely run`이 자기 작업 목록을 끝까지 돌리고, 머신 공용 장부로 서로 조율하는 구조입니다. 서브시스템으로 키우려면 이런 것들이 다음 후보입니다.

- **데몬 + `tablely submit`**: 이미 돌고 있는 실행에 작업을 추가하거나 취소. 지금은 새 작업 목록마다 `tablely run`을 하나 더 띄워야 합니다.
- **선점(preemption) 확대**: 지금은 `switchable` 작업만 체크포인트 후 CPU로 옮기는 방식으로 GPU를 내줍니다. GPU 전용 작업을 멈췄다가 나중에 재개하는 일반 선점은 아직 없습니다.
- **실제 GPU 서버 검증**: `tablely.stream`의 CUDA 경로(고정 메모리, 비동기 복사, 다중 GPU `map_chunks`)는 GPU가 없는 환경에서 만들었습니다. 파이프라인 로직은 대역(가짜 backend)으로 테스트했지만 실제 GPU에서는 아직 돌려보지 않았습니다.
- **GPU 나눠 쓰기**: 작은 작업 여러 개가 한 GPU를 공유 (MPS, MIG, 메모리 비율 제한).
- **토폴로지 인지**: GPU와 같은 NUMA 노드의 코어를 우선 배정.
- **사용률 기반 조정**: 실제 GPU/CPU 사용률을 보고 코어를 재분배. 예를 들어 GPU 사용률이 낮으면 데이터 로딩 코어를 늘립니다.
- **스레드 수 동적 조정**: 지금은 코어가 바뀌어도 `OMP_NUM_THREADS`는 시작 시점 값 그대로라서, 스크립트가 `client.sync_torch_threads()`를 불러야 합니다.
- AMD ROCm GPU 자동 감지 (지금은 `--gpus`로 직접 지정해야 함).
