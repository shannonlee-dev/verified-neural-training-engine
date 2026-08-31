# Verified Neural Training Engine 설계

## 목적

NumPy만으로 동적 계산 그래프 기반 미니 딥러닝 엔진을 구현하고, 수치 미분과 학습 실험으로 구현 정확성과 초기화 전략의 영향을 증명한다. 필수 데이터는 XOR와 MNIST로 한정하며 외부 ML/DL 프레임워크는 사용하지 않는다.

## 범위

### 포함

- `Tensor`와 동적 AutoGrad 계산 그래프
- 사칙연산, 행렬 곱과 학습에 필요한 축약·비선형 Tensor 연산
- `Linear`, `ReLU`, `Sigmoid`, `Softmax`
- `SGD`, `Adam`, `zero_grad()`
- Zero, Random, He, Xavier 초기화
- 주요 Tensor 연산과 모든 필수 레이어의 Gradient Check
- XOR 학습과 Zero 초기화 실패 실험
- MNIST IDX gzip 다운로드, 직접 파싱, 캐시 및 1 epoch 학습
- 초기화별 손실 비교 그래프
- 단위 테스트, 실행 로그, 검증·실험 리포트와 README

### 제외

- `load_digits`, `make_moons`, CIFAR-10
- PyTorch, TensorFlow, Keras, torchvision 및 자동 데이터 로더
- 보너스 과제인 Dropout, L2 정규화와 추가 활성화 함수

## 구조

```text
neural_engine/
├── core/
│   ├── tensor.py
│   └── utils.py
├── nn/
│   ├── module.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   └── initialization.py
├── optim/
│   ├── optimizer.py
│   ├── sgd.py
│   └── adam.py
└── data/
    └── mnist.py
scripts/
├── gradient_check.py
├── train_xor.py
├── train_mnist.py
└── compare_initialization.py
tests/
reports/
├── verification_report.md
└── experiment_report.md
logs/
figures/
README.md
```

## AutoGrad 설계

`Tensor`는 `data`, `grad`, `requires_grad`, 부모 노드, 연산 이름과 `_backward` 함수를 가진다. 순전파 연산은 결과 Tensor와 로컬 역전파 함수를 동적으로 생성한다.

`Tensor.backward()`는 현재 Tensor에서 깊이 우선 탐색으로 위상 순서를 만든 뒤, 출력 기울기를 설정하고 노드를 역순으로 순회하며 각 `_backward()`를 실행한다. 동일 Tensor로 들어오는 여러 경로의 기울기는 누적하여 Chain Rule을 적용한다.

NumPy broadcasting으로 확장된 피연산자의 기울기는 `sum_to_shape(gradient, original_shape)`로 원래 차원에 맞게 합산한다. 선행 차원을 먼저 합산하고, 크기가 1이었던 축은 `keepdims=True`로 합산한다.

학습과 손실 계산에 필요한 연산은 사칙연산, 부호 반전, 거듭제곱, 행렬 곱, 합계, 평균, 지수, 로그와 인덱싱을 제공한다. 각 연산은 역전파 규칙과 브로드캐스팅 복원을 포함한다.

## 신경망 API

`Module`은 중첩 모듈에서 `parameters()`를 수집하고 학습 가능한 Tensor를 노출한다. `Linear`는 입력·출력 크기와 초기화 이름을 받아 weight와 bias를 만든다. Bias는 영으로 초기화하고 초기화 비교는 weight 전략만 변경한다.

활성화 레이어는 Tensor 입력을 받아 Tensor를 반환한다. Softmax는 최대값을 빼는 안정화 후 확률을 계산하고 일반적인 Jacobian-vector product로 역전파한다. 분류 학습에는 수치 안정성을 위해 logits 기반 Cross Entropy를 사용하며, Softmax 레이어 자체도 독립 Gradient Check 대상이다.

초기화 표준편차는 다음과 같다.

- Zero: `0`
- Random: `1`
- He: `sqrt(2 / n_in)`
- Xavier: `sqrt(2 / (n_in + n_out))`

## 최적화와 학습 순서

`SGD`는 선택적 momentum 없이 기본 경사하강을 구현한다. `Adam`은 1차·2차 모멘트와 bias correction을 적용한다. 두 옵티마이저 모두 `step()`과 `zero_grad()`를 제공한다.

모든 학습 스텝의 순서는 `optimizer.zero_grad()` → 순전파 → 손실 계산 → `loss.backward()` → `optimizer.step()`이다. 이 순서를 스크립트와 README 예제에 동일하게 사용하여 이전 스텝의 기울기가 섞이지 않게 한다.

모든 실행 진입점은 기본 seed를 하나의 상수에서 가져오며 CLI `--seed`로 덮어쓸 수 있다. 데이터 셔플, 초기화와 실험은 같은 NumPy 난수 생성기 계열을 사용한다.

## 검증 설계

중앙 차분 `(f(x + 1e-5) - f(x - 1e-5)) / (2e-5)`으로 수치 기울기를 구한다. 상대오차는 `max(abs(a-n) / max(1e-12, abs(a)+abs(n)))`로 계산하고 모든 항목이 `1e-7` 이하일 때만 스크립트가 성공한다.

검증 대상은 덧셈·곱셈·나눗셈·행렬 곱·합계/평균과 Linear·ReLU·Sigmoid·Softmax이다. 브로드캐스팅 입력을 포함하며 Linear의 입력, weight와 bias를 각각 확인한다. 스크립트는 항목별 최대 상대오차와 최종 통과 여부를 터미널 및 `logs/gradient_check.log`에 기록한다.

단위 테스트는 계산 그래프 누적, 브로드캐스팅 축소, 옵티마이저 갱신, 초기화 분산, IDX 파싱 및 gradient check 허용오차를 다룬다.

## XOR 및 초기화 실험

XOR 모델은 `2 → hidden → 1` MLP, ReLU, Sigmoid/Binary Cross Entropy로 구성한다. 동일한 모델·seed·optimizer에서 initialization 옵션만 바꾼다.

- He: 100 epoch 안에 loss `< 0.1` 또는 accuracy `>= 95%`
- Zero: 50 epoch 뒤에도 대칭성 때문에 정확도가 XOR 해결 기준에 도달하지 못하고 손실 감소가 미미함
- 비교 그래프: Zero, Random, He의 epoch별 loss를 한 그림에 표시

`train_xor.py`는 초기화와 epoch를 CLI 인자로 받고 학습 로그를 저장한다. `compare_initialization.py`는 세 전략을 동일 조건으로 실행해 `figures/initialization_loss.png`, 원시 수치 로그와 요약을 만든다.

## MNIST 데이터 및 학습

`train_mnist.py` 실행 시 네 IDX gzip 파일이 없으면 `urllib`로 다운로드하여 `data/`에 저장한다. `gzip`으로 읽은 바이트의 magic number, dtype, 차원과 크기를 검증하고 `struct`로 헤더를 직접 파싱하여 NumPy 배열로 변환한다. 다음 실행은 캐시를 재사용한다. 원본 데이터는 `.gitignore`로 제외한다.

이미지는 `float32`로 변환해 `[0, 1]`로 정규화하고 평탄화한다. 모델은 `784 → 128 → 10`, ReLU, logits Cross Entropy이며 Adam과 직접 구현한 mini-batch iterator를 사용한다. 기본 실행은 전체 훈련 데이터 1 epoch 후 테스트 정확도 80% 이상을 목표로 하고 epoch별 loss·accuracy를 `logs/mnist.log`에 기록한다.

다운로드 오류, 잘못된 IDX 헤더와 크기 불일치는 원인을 포함한 예외로 중단한다. 테스트는 작은 인메모리 gzip/IDX fixture로 파서를 검증하므로 네트워크에 의존하지 않는다.

## 산출물과 재현성

`requirements.txt`에는 NumPy와 Matplotlib만 둔다. README는 프로젝트 개요, 설치, 실행 순서, 구조, Tensor와 학습 API 예제, seed 사용법, 결과 요약, 평가 체크리스트를 포함한다.

검증 리포트는 AutoGrad, 중앙 차분, 상대오차, Gradient Check를 학습과 분리하는 이유와 실제 결과를 설명한다. 실험 리포트는 Zero 대칭성, Random의 분산, ReLU와 He의 관계, Xavier 용도, Adam 원리 및 XOR/MNIST/초기화 비교 결과를 분석한다.

최종 저장 산출물은 다음과 같다.

- `logs/gradient_check.log`
- `logs/xor_he.log`, `logs/xor_zero.log`, `logs/mnist.log`
- `logs/initialization_comparison.csv`
- `figures/initialization_loss.png`
- `reports/verification_report.md`
- `reports/experiment_report.md`

## 완료 조건

- 주요 Tensor 연산 및 Linear, ReLU, Sigmoid, Softmax의 상대오차 `<= 1e-7`
- He 초기화 XOR가 100 epoch 안에 loss `< 0.1` 또는 accuracy `>= 95%`
- Zero 초기화 XOR가 50 epoch 뒤에도 성공 기준에 미달
- Zero, Random, He 손실 비교 그래프 생성
- MNIST 1 epoch 테스트 정확도 `>= 80%`
- 자동 테스트 통과 및 모든 명령을 README만으로 재현 가능

