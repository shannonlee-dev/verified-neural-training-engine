# Verified Neural Training Engine

[![CI](https://github.com/shannonlee-dev/verified-neural-training-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/shannonlee-dev/verified-neural-training-engine/actions/workflows/ci.yml)

## 프로젝트 소개

NumPy만으로 구현한 동적 계산 그래프 기반 미니 딥러닝 프레임워크입니다. Tensor 연산이 순전파 중 계산 그래프를 만들고, `backward()`가 위상 정렬된 그래프를 역순으로 순회하며 Chain Rule로 기울기를 전달합니다.

구현의 정확성은 중앙 차분 Gradient Check로 검증하고, XOR와 MNIST 학습 및 Zero/Random/He 초기화 비교 실험으로 실제 수렴 여부를 확인합니다. PyTorch, TensorFlow, Keras, torchvision과 자동 데이터 로더는 사용하지 않습니다.

## 핵심 특징

- Tensor 사칙연산, 행렬 곱, 축약, 지수·로그와 동적 AutoGrad
- broadcasting 역전파를 위한 `sum_to_shape()`
- Linear, ReLU, Sigmoid, Softmax와 안정적인 logits Cross Entropy
- SGD와 bias correction을 포함한 Adam
- Zero, Random, He, Xavier 초기화
- 주요 Tensor 연산 및 모든 필수 레이어 Gradient Check
- XOR He 성공과 Zero 50 epoch 실패 재현
- MNIST IDX gzip 자동 다운로드, 직접 바이너리 파싱과 캐시
- `no_grad()`를 통한 추론 시 계산 그래프 기록 중단
- 고정 seed, 실행 로그, 비교 CSV·그래프와 검증·실험 리포트

## 아키텍처

```text
NumPy ndarray
    ↓
Tensor + 동적 계산 그래프 ── sum_to_shape() broadcasting 복원
    ↓
Module / Linear / ReLU / Sigmoid / Softmax / Loss
    ↓
SGD / Adam ── zero_grad() → forward → backward() → step()
    ↓
XOR 실험 · MNIST 학습 · Gradient Check
    ↓
logs/ · figures/ · reports/
```

`core`는 NumPy에만 의존하고, `nn`은 Tensor 연산을 조합합니다. `optim`은 `Module.parameters()`가 반환한 학습 Tensor만 갱신합니다. 데이터 로더와 실행 스크립트는 이 세 계층의 공개 API를 사용합니다.

## 설치 방법

Python 3.10 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

의존성은 NumPy와 그래프 출력용 Matplotlib뿐입니다. 테스트는 Python 표준 `unittest`를 사용합니다.

## 실행 방법

저장소 루트에서 다음 순서로 실행합니다.

```bash
# 1. 전체 단위 테스트
python3 -m unittest discover -s tests -v

# 2. Tensor 연산과 필수 레이어 Gradient Check
python3 scripts/gradient_check.py

# 3. XOR: He 성공과 Zero 실패 재현
python3 scripts/train_xor.py --initialization he --epochs 100 --log-file logs/xor_he.log
python3 scripts/train_xor.py --initialization zero --epochs 50 --log-file logs/xor_zero.log

# 4. Zero / Random / He 비교 CSV와 그래프 생성
python3 scripts/compare_initialization.py

# 5. MNIST 전체 데이터 1 epoch 학습 (정확도 목표 95%)
python3 scripts/train_mnist.py --epochs 1 --log-file logs/mnist.log
```

MNIST 최초 실행은 네 개의 표준 IDX gzip 파일을 `data/`에 내려받습니다. `gzip` 압축을 읽고 IDX magic number·차원·payload 크기를 직접 검증해 NumPy 배열로 변환합니다. 이후 실행은 캐시를 재사용하며 `data/*.gz`는 Git에서 제외됩니다.

빠른 파이프라인 확인에는 제한 옵션을 사용할 수 있습니다. 제한 실행은 95% 완료 기준 판정에서 제외됩니다.

```bash
python3 scripts/train_mnist.py --train-limit 1000 --test-limit 500
```

모든 CLI의 기본 seed는 `42`이며 `--seed`로 변경할 수 있습니다. MNIST CLI와 Python API의 기본 모델은 `784 → 256 → 10`이며, `--hidden-features` 또는 `hidden_features`로 은닉층 크기를 변경할 수 있습니다.
라이브러리에서는 모델 빌더의 `seed` 또는 `Linear`의 `rng` 인자로 난수 상태를 명시할 수 있으며, 같은 seed로 빌드한 모델은 같은 파라미터로 초기화됩니다.

## 사용 예시

### Tensor와 AutoGrad

```python
from neural_engine import Tensor

x = Tensor([[1.0, 2.0]], requires_grad=True)
w = Tensor([[3.0], [4.0]], requires_grad=True)
loss = (x @ w).mean()
loss.backward()

print(x.grad)  # [[3. 4.]]
print(w.grad)  # [[1.], [2.]]
```

`backward()`는 leaf gradient를 누적합니다. 같은 그래프에서 두 번 호출하면 기울기도 두 번 누적됩니다. 학습에서는 이전 step의 기울기를 반드시 먼저 초기화해야 합니다.

추론처럼 새 계산 그래프가 필요하지 않은 경우에는 `no_grad()`를 사용합니다.

```python
from neural_engine import no_grad

with no_grad():
    logits = model(inputs)
```

`no_grad()`는 컨텍스트 안에서 새로 만들어지는 연산 결과의 그래프 기록을 중단하고, 컨텍스트가 끝나면 원래 상태로 돌아갑니다. 파라미터 자체의 `requires_grad`는 변경하지 않으며, `train()`·`eval()`이나 Dropout 제어를 제공하는 API는 아닙니다.

### 모델 학습 순서

```python
import numpy as np

from neural_engine import Tensor
from neural_engine.nn import Linear, ReLU, Sequential, cross_entropy
from neural_engine.optim import Adam

model = Sequential(Linear(2, 8, "he"), ReLU(), Linear(8, 2, "he"))
optimizer = Adam(model.parameters(), lr=0.01)

inputs = Tensor([[0.0, 1.0], [1.0, 0.0]])
targets = np.array([1, 0])

optimizer.zero_grad()       # 1. 이전 gradient 초기화
logits = model(inputs)      # 2. forward 및 동적 그래프 생성
loss = cross_entropy(logits, targets)
loss.backward()             # 3. 위상 정렬 후 역순 Chain Rule
optimizer.step()            # 4. 파라미터 갱신
```

## 모듈 구조

```text
neural_engine/
├── core/                   # Tensor, AutoGrad, broadcasting gradient 축소
├── nn/                     # Module, 레이어, 활성화, 손실, 초기화
├── optim/                  # SGD, Adam, zero_grad
├── data/mnist.py           # IDX gzip 다운로드·직접 파싱·배치
├── experiments.py          # 공통 XOR 학습
├── mnist_training.py       # MNIST 모델·학습·평가
└── verification.py         # 중앙 차분과 Gradient Check
scripts/                    # 재현 가능한 CLI 진입점
tests/                      # unittest 단위·통합 테스트
logs/                       # 실제 검증·학습 기록
figures/                    # 초기화 Loss 비교 그래프
reports/                    # 검증 및 실험 분석
```

## 구현 원리

### 계산 그래프와 Chain Rule

각 결과 Tensor는 부모 노드와 로컬 `_backward()` 함수를 저장합니다. `Tensor.backward()`는 출력부터 DFS로 위상 순서를 구성하고, 출력 기울기를 설정한 다음 역순으로 `_backward()`를 호출합니다. 한 Tensor로 들어오는 여러 경로의 기울기는 더해집니다.

### Broadcasting 역전파

순전파에서 크기 1인 축이나 선행 축이 확장되면, 역전파 기울기는 원래 피연산자 shape로 돌아가야 합니다. `sum_to_shape()`는 추가된 선행 축을 합산하고 크기 1이었던 축을 `keepdims=True`로 합산합니다.

### Gradient Check

수치 기울기는 epsilon `1e-5`의 중앙 차분으로 계산합니다.

```text
grad ≈ (f(x + ε) - f(x - ε)) / (2ε)
relative error = |analytic - numerical| / max(1e-12, |analytic| + |numerical|)
```

수치 미분은 파라미터마다 순전파를 반복하므로 학습에는 비효율적입니다. 대신 역전파 구현을 학습 전에 독립적으로 검증하는 기준으로 사용합니다. ReLU는 0에서 미분 불가능하므로 검증 입력에서 0을 제외합니다.

### 초기화와 옵티마이저

| 초기화 | weight 표준편차 | 용도 |
|---|---:|---|
| Zero | `0` | 대칭성 실패 재현 |
| Random | `1` | 기본 표준정규 비교군 |
| He | `sqrt(2 / n_in)` | ReLU 신호 분산 유지 |
| Xavier | `sqrt(2 / (n_in + n_out))` | Sigmoid/Tanh 계열 분산 유지 |

Adam은 Momentum 계열의 1차 모멘트 `m`과 RMSProp 계열의 2차 모멘트 `v`를 결합하며, 두 모멘트에 bias correction을 적용합니다.

## 검증된 결과

기본 seed `42`에서 생성된 저장 로그 기준입니다.

| 검증 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| 최대 Gradient Check 상대오차 | `1.049e-10` | `<= 1e-7` | PASS |
| XOR He, 100 epoch loss | `0.015496` | `< 0.1` | PASS |
| XOR He, 100 epoch accuracy | `100%` | `>= 95%` | PASS |
| XOR Zero, 50 epoch loss | `0.693147` | 감소 없음 | 실패 재현 |
| XOR Zero, 50 epoch accuracy | `50%` | `< 95%` | 실패 재현 |
| MNIST, 1 epoch test accuracy | `95.21%` | `>= 95%` | PASS |

상세 수치는 `reports/verification_report.md`, `reports/experiment_report.md`와 `logs/`에서 확인할 수 있습니다.

## 미션 완료 체크리스트

- [x] Tensor `data`, `grad`, `_backward`와 동적 계산 그래프
- [x] 위상 정렬 → 역순 순회 → Chain Rule 역전파
- [x] 사칙연산, MatMul과 broadcasting gradient 복원
- [x] Linear, ReLU, Sigmoid, Softmax
- [x] SGD, Adam과 명시적 `zero_grad()`
- [x] Zero, Random, He, Xavier 초기화
- [x] 모든 필수 연산·레이어 Gradient Check `<= 1e-7`
- [x] He 초기화 XOR 100 epoch 이내 성공
- [x] Zero 초기화 XOR 50 epoch 실패 재현
- [x] Zero/Random/He Loss 비교 CSV와 `figures/initialization_loss.png`
- [x] IDX gzip 직접 파싱 MNIST 학습 스크립트와 로그
- [x] MNIST 1 epoch 정확도 95% 이상
- [x] 고정 seed, 설치·실행 방법, 검증·실험 리포트
