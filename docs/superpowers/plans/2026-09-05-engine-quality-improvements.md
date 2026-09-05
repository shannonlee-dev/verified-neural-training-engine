# Engine Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 현재 요청은 계획 문서 작성 후 대기이므로, 별도 구현 지시 전에는 실행하지 않는다.

**Goal:** MNIST 추론의 구조 결합과 기본값 불일치를 해소하고, 초기화 실험의 해석과 두 가지 수치 계산 오류를 보완한다.

**Architecture:** 기존 `core → nn/optim → 학습·검증` 경계를 유지한다. `core`에 작은 `no_grad()` 실행 컨텍스트를 추가하고 MNIST 추론은 기존 모델의 순전파를 재사용한다. 설정·문서·수치 오류는 각각 해당 모듈 안에서 수정한다.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, 표준 라이브러리 `contextlib`, `contextvars`, `unittest`.

**Spec:** 사용자가 2026-09-05 요청한 네 가지 진단 항목이 변경 범위다. 과제 조건은 `docs/private/mission.md`, `docs/private/rubric.md`, 기존 구조 설명은 `docs/superpowers/specs/2026-08-31-neural-training-engine-design.md`를 참조한다. 과거 설계 문서의 MNIST 기본값 128보다 현재 README·CLI의 256을 이 계획의 기준으로 삼는다.

## 상태와 제약

- 상태: 계획 작성 완료, 구현 대기. 아래 체크박스는 모두 미래 구현 작업이다.
- 현재 작업에서는 소스 코드, 테스트, 기존 로그·그래프를 변경하지 않는다. 커밋·푸시하지 않는다.
- 모든 학습·자동미분 계산은 NumPy로 구현하며 외부 ML/DL 프레임워크를 사용하지 않는다.
- Gradient Check 중앙 차분 epsilon은 `1e-5`, 상대오차 기준은 `<= 1e-7`이다.
- Zero, Random, He, Xavier 초기화와 seed `42`를 유지한다.
- 기존 `zero_grad → forward → loss → backward → step` 동작과 leaf gradient 누적 규칙을 유지한다.
- MNIST IDX 직접 파싱과 캐시, CSV 열 형식, 기존 CLI 옵션을 유지한다.
- 과제의 MNIST 최소 기준은 80%다. 현재 프로젝트의 기본 전체 실행 목표인 95%도 유지한다.
- 범용 Trainer, 모델 저장, Dropout, dtype 정책 변경, 폴더 재배치는 범위 밖이다.
- 구현은 현재 작업 트리 상태를 먼저 확인하고 순차 진행한다. 별도 승인 없이 이 문서를 구현·커밋 지시로 해석하지 않는다.

## 진단 시 확보한 기준값

2026-09-05 변경 전 직접 실행한 결과다. 변경 후 검증 결과로 재사용하지 않는다.

| 항목 | 결과 |
|---|---|
| 전체 unittest | 43개 통과 |
| Gradient Check | 13개 통과, 최대 상대오차 `1.049e-10` |
| XOR He 100 epoch | loss `0.015496`, accuracy `100%` |
| XOR Zero 50 epoch | loss `0.693147`, accuracy `50%`, loss 정체 |
| 전체 MNIST 1 epoch, hidden 256 | loss `0.305141`, test accuracy `95.21%` |
| 거듭제곱 경계값 | `x=0`에서 `x**0`의 gradient가 `NaN` |
| 수치 미분 예외 | 평가 함수가 실패하면 입력 `[1.0]`이 `[1.00001]`로 남음 |

## 변경 파일과 책임

| 파일 | 변경 책임 |
|---|---|
| `neural_engine/core/grad_mode.py` (신규) | 중첩·예외 복구를 지원하는 gradient 기록 상태 |
| `neural_engine/core/tensor.py` | 연산 결과의 그래프 기록 제어, 0제곱 역전파 수정 |
| `neural_engine/core/__init__.py`, `neural_engine/__init__.py` | `no_grad` 공개 export |
| `neural_engine/nn/activations.py`, `neural_engine/nn/losses.py` | 비기록 결과에 backward closure를 저장하지 않음 |
| `neural_engine/mnist_training.py` | 모델 순전파 기반 추론, 기본 은닉층 크기 통일 |
| `neural_engine/config.py`, `scripts/train_mnist.py` | MNIST 은닉층 기본값의 단일 정의 |
| `neural_engine/verification.py` | 수치 미분 도중 예외가 발생해도 입력 복원 |
| `tests/test_tensor.py`, `tests/test_nn.py` | 그래프 기록 제어와 거듭제곱 회귀 검사 |
| `tests/test_mnist.py` | 다양한 모델의 추론, 기본 설정 경로 검사 |
| `tests/test_gradient_check.py` | 수치 미분의 성공·실패 시 입력 복원 검사 |
| `README.md` | `no_grad` 사용법과 MNIST 기본 설정 명시 |
| `reports/experiment_report.md` | 초기화 실험에서 확인한 사실과 해석 한계 명시 |

## Task 1: 모델 순전파를 재사용하는 추론

**Interfaces**

- 신규 공개 API: `from neural_engine import no_grad`, `with no_grad(): ...`.
- 내부 API: `is_grad_enabled() -> bool`.
- 변경 API: `predict_mnist(model: Module, inputs: np.ndarray, batch_size: int = 1024) -> np.ndarray`.
- 모델 출력 계약: `(batch, classes)` 형태의 Tensor. 반환값은 `int64` 클래스 인덱스다.

**설계 선택:** 레이어별 NumPy 추론 구현을 추가하는 방식은 계산 중복을 늘린다. 파라미터의 `requires_grad`를 일시 변경하는 방식은 모델 상태를 건드리고 중첩 호출에 취약하다. 컨텍스트 상태로 새 연산의 그래프 기록만 끄는 방식을 사용한다.

- [ ] **1. 비기록 실행의 회귀 테스트를 먼저 추가한다.**

`tests/test_tensor.py`에 다음 핵심 사례를 추가한다. `no_grad`는 공개 export에서 import한다.

```python
def test_no_grad_detaches_results_and_restores_tracking(self):
    x = Tensor([2.0], requires_grad=True)
    with no_grad():
        y = x * x
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad)
        self.assertEqual(y._prev, ())
        self.assertIsNone(y._backward.__closure__)
    self.assertTrue(x.requires_grad)
    (x * x).sum().backward()
    np.testing.assert_array_equal(x.grad, [4.0])

def test_no_grad_restores_outer_state_after_exception(self):
    x = Tensor([2.0], requires_grad=True)
    with no_grad():
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with no_grad():
                raise RuntimeError("probe")
        self.assertFalse((x + 1).requires_grad)
    self.assertTrue((x + 1).requires_grad)
```

추가 사례로 컨텍스트 안에서 명시적으로 생성한 leaf `Tensor(..., requires_grad=True)`는 지정값을 유지하되, 해당 leaf의 연산 결과는 그래프를 기록하지 않는지 검사한다. 기존 leaf의 데이터와 gradient가 추론 전후에 변하지 않는지도 확인한다.

- [ ] **2. 모델 구조에 의존하지 않는 추론 회귀 테스트를 추가한다.**

`tests/test_mnist.py`의 고정 구조 테스트는 기존 예측값 검사를 유지하면서 이름을 변경한다. 다음 사례도 추가한다.

```python
def test_predict_accepts_alternative_model_and_disables_graph(self):
    class RecordingModel(Module):
        def __init__(self):
            self.layer = Linear(2, 2, initialization="zero")
            self.layer.weight.data[:] = np.eye(2)
            self.outputs = []

        def forward(self, inputs):
            output = Sigmoid()(self.layer(inputs))
            self.outputs.append(output)
            return output

    model = RecordingModel()
    inputs = np.array([[3.0, -1.0], [-2.0, 4.0], [2.0, 1.0]])
    expected = np.argmax(model(Tensor(inputs)).data, axis=1)
    model.outputs.clear()
    actual = predict_mnist(model, inputs, batch_size=2)
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.dtype, np.int64)
    self.assertEqual(len(model.outputs), 2)
    for output in model.outputs:
        self.assertFalse(output.requires_grad)
        self.assertIsNone(output.grad)
        self.assertEqual(output._prev, ())
        self.assertIsNone(output._backward.__closure__)
```

이 테스트에 필요한 `Tensor`, `Module`, `Sigmoid` import를 추가한다.
별도 사례로 `Sequential(Linear(2, 3), ReLU(), Linear(3, 2), Softmax())`에서도 일반 forward와 클래스 예측이 일치하는지 확인한다.

- [ ] **3. 테스트를 실행해 변경 전 실패를 확인한다.**

```bash
python3 -m unittest tests.test_tensor tests.test_mnist -v
```

예상 실패는 `no_grad` 미구현과 현행 `predict_mnist`의 고정 구조 검사다.

- [ ] **4. 기록 상태와 Tensor의 연동을 구현한다.**

신규 `neural_engine/core/grad_mode.py`:

```python
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_grad_enabled: ContextVar[bool] = ContextVar("grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


@contextmanager
def no_grad() -> Iterator[None]:
    token = _grad_enabled.set(False)
    try:
        yield
    finally:
        _grad_enabled.reset(token)
```

`Tensor.__init__`에서는 `_parents`가 있는 경우에만 기록 상태를 반영한다. 명시적인 leaf 생성의 의미는 바꾸지 않는다.

```python
self.requires_grad = bool(requires_grad) and (
    not _parents or is_grad_enabled()
)
self.grad = np.zeros_like(self.data) if self.requires_grad else None
self._prev = tuple(_parents) if self.requires_grad else ()
```

`core/tensor.py`, `nn/activations.py`, `nn/losses.py`의 모든 `output._backward = backward`를 다음과 같이 변경한다. `_prev`에서 부모를 제거해도 closure가 부모를 보관할 수 있으므로 이 처리도 필요하다.

```python
if output.requires_grad:
    output._backward = backward
```

비기록 Tensor는 기존의 아무 작업도 하지 않는 `_backward`를 유지한다. 두 `__init__.py`에서 `no_grad`를 export하고 `__all__`도 갱신한다.

- [ ] **5. `predict_mnist`를 모델의 forward에 연결한다.**

`Module`과 공개 `no_grad`를 import한다. 고정 레이어 수·타입 검사, weight/bias 직접 읽기, 별도로 작성한 Linear/ReLU 계산을 삭제한다.

```python
def predict_mnist(model: Module, inputs: np.ndarray,
                  batch_size: int = 1024) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    predictions = []
    with no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            logits = model(Tensor(batch))
            if logits.ndim != 2 or logits.shape[0] != len(batch) or logits.shape[1] == 0:
                raise ValueError("prediction expects shape (batch, classes)")
            predictions.append(np.argmax(logits.data, axis=1))
    if not predictions:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64, copy=False)
```

- [ ] **6. 비기록 동작과 모델 상태 보존을 검증한다.**

`subTest`로 `+`, `-`, unary `-`, `*`, `/`, `**`, `@`, `sum`, `mean`, `exp`, `log`, `reshape`, indexing, ReLU, Sigmoid, Softmax, BCE, Cross Entropy 결과를 확인한다. 올바른 정의역의 입력을 사용하고 결과의 `requires_grad=False`, `grad=None`, `_prev=()`, backward closure 미보관을 검사한다. `no_grad` 밖에서는 기존 gradient 검사가 통과하는지 확인한다.

빈 입력에는 빈 `int64` 배열을 반환하고, 잘못된 batch size와 출력 shape에는 `ValueError`가 발생하는지 검사한다. 학습한 파라미터의 data와 grad를 추론 전에 복사하고, 추론 후 배열이 같은지 검사한다.

```bash
python3 -m unittest tests.test_tensor tests.test_nn tests.test_mnist tests.test_gradient_check -v
```

- [ ] **7. README에 공개 API와 제한을 설명한다.**

사용 예시에 `with no_grad(): logits = model(inputs)`를 추가한다. “새로운 연산 결과의 그래프 기록을 중단하고 컨텍스트 종료 시 원래 상태로 돌아간다. 파라미터 자체의 `requires_grad`는 변경하지 않는다”고 설명한다. `train()/eval()`이나 Dropout 제어를 제공하는 API로 설명하지 않는다.

**완료 기준:** 층수·활성화가 다른 모델로 추론할 수 있고 기존 forward와 결과가 일치한다. 추론 결과가 부모·backward closure를 보관하지 않으며, 추론 후 학습과 gradient 누적이 기존대로 동작한다.

## Task 2: MNIST 기본값을 256으로 통일

**Interfaces:** `neural_engine.config.DEFAULT_MNIST_HIDDEN_FEATURES: int = 256`을 유일한 기본값 정의로 두고 모델 빌더·학습 함수·CLI가 참조한다.

- [ ] **1. 호출 경로의 회귀 테스트를 추가한다.**

`tests/test_mnist.py`에서 모델 빌더를 감싸는 mock을 사용해 극소량 데이터로 `train_mnist`를 은닉층 인자 생략 상태에서 1 epoch 실행한다. 실제 빌더로 전달된 `hidden_features`가 256인지 검사한다.

```python
with patch("neural_engine.mnist_training.build_mnist_model",
           wraps=build_mnist_model) as builder:
    train_mnist(inputs, targets, inputs, targets,
                epochs=1, batch_size=2, class_count=2)
self.assertEqual(builder.call_args.kwargs["hidden_features"], 256)
self.assertEqual(build_mnist_model().layers[0].out_features, 256)
```

`inputs=np.eye(2, dtype=np.float32)`, `targets=np.array([0, 1])`을 사용한다. `unittest.mock.patch`와 `build_mnist_model`을 import한다. 기존 CLI gzip fixture 테스트를 이용해 `scripts.train_mnist.train_mnist`를 `wraps=train_mnist`로 관측하고, `--hidden-features` 생략 시 256, 명시적 `4` 지정 시 4가 전달되는지 검사한다.

- [ ] **2. `python3 -m unittest tests.test_mnist -v`를 실행해 기존 기본값 128로 인한 실패를 확인한다.**
- [ ] **3. `config.py`에 상수를 추가하고 빌더·함수·CLI의 기본값을 교체한다.**

```python
DEFAULT_MNIST_HIDDEN_FEATURES = 256

# build_mnist_model / train_mnist 인자
hidden_features: int = DEFAULT_MNIST_HIDDEN_FEATURES

# CLI
parser.add_argument("--hidden-features", type=int,
                    default=DEFAULT_MNIST_HIDDEN_FEATURES)
```

다른 값까지 이번 작업에 맞춰 설정 계층으로 이동하지 않는다.

- [ ] **4. 같은 테스트를 재실행하고 README에 “CLI·Python API 공통 기본 모델은 784 → 256 → 10”이라고 명시한다.**

**완료 기준:** 기본값이 세 경로에서 256이고 명시적으로 지정한 값이 우선한다. 기존 로그 형식과 CLI 옵션은 유지된다.

## Task 3: 초기화 실험 해석 수정

**Files:** `reports/experiment_report.md`.

- [ ] **1. 첫 층과 출력층의 관계를 코드에서 재확인한다.**

```bash
python3 - <<'PY'
import numpy as np
from neural_engine.experiments import build_xor_model
random = build_xor_model("random", seed=42)
he = build_xor_model("he", seed=42)
np.testing.assert_array_equal(random.layers[0].weight.data, he.layers[0].weight.data)
np.testing.assert_array_equal(he.layers[2].weight.data, random.layers[2].weight.data * 0.5)
print("First layer identical; He output weights are half of Random.")
PY
```

- [ ] **2. “Random과 He” 절에 다음 설명을 추가한다.**

> 이 XOR 모델의 첫 층은 입력 차원이 2이므로 He 표준편차가 `sqrt(2 / 2) = 1`로 Random과 같다. 동일 seed에서는 첫 층의 초기 weight 자체도 동일하다. 두 조건의 차이는 Sigmoid에 연결되는 출력층에 있으며, He weight는 Random의 0.5배다. 따라서 이 비교는 해당 모델에서 초기화 조건에 따라 수렴 차이가 발생했음을 보여주지만, ReLU 층에서 He의 분산 유지 효과를 단독으로 입증하지는 않는다. 또한 결과는 seed 42의 단일 조건이므로 초기화 방법의 일반적인 우열을 결론짓지 않는다.

He의 일반적인 설계 의도와 이 실험에서 관측한 사실을 구분한다.

- [ ] **3. 결론 절을 다음 내용으로 교체한다.**

> 검증 대상 연산·레이어가 Gradient Check를 통과했고, He 조건의 XOR와 MNIST 학습 성공, Zero 조건의 XOR loss 정체를 재현했다. Zero 조건에서는 대칭성과 함께 ReLU의 0 지점 도함수를 0으로 정의한 구현이 hidden weight의 업데이트 정지에 관여한다. Random과 He의 수렴 차이는 앞서 명시한 모델·seed·출력층 초기 스케일 조건에서 관측한 결과다. Gradient Check 통과는 검사한 입력에서의 정확성을 뒷받침하며, 모든 입력·연산에 오류가 없음을 증명하지는 않는다.

기존 “초기 loss” 표현은 `Epoch 1 loss`와 일치시키고, 초기화 직후·업데이트 전 측정값과 혼동하지 않도록 수정한다.

- [ ] **4. 기존 표·CSV·그래프를 다시 읽고 수치 변경 없이 해석만 수정됐는지 확인한다.**

**완료 기준:** 첫 층이 동일하다는 사실, 출력층의 차이, 단일 seed의 한계가 명시된다. 새로운 실험이나 여러 seed 검증을 수행했다고 기록하지 않는다. 문서 변경만을 위한 자동 테스트는 추가하지 않는다.

## Task 4: 0제곱 역전파의 NaN 수정

**Files:** `neural_engine/core/tensor.py`, `tests/test_tensor.py`.

- [ ] **1. 0제곱 회귀 테스트를 추가하고 실패를 확인한다.**

```python
def test_zero_power_has_zero_gradient_at_zero(self):
    x = Tensor([0.0, -2.0, 3.0], requires_grad=True)
    with np.errstate(divide="raise", invalid="raise"):
        y = x ** 0
        np.testing.assert_array_equal(y.data, np.ones(3))
        y.backward(np.array([2.0, -1.0, 4.0]))
    np.testing.assert_array_equal(x.grad, np.zeros(3))

def test_zero_power_preserves_accumulated_leaf_gradient(self):
    x = Tensor([0.0], requires_grad=True)
    x.sum().backward()
    (x ** 0).sum().backward()
    np.testing.assert_array_equal(x.grad, [1.0])
```

```bash
python3 -m unittest tests.test_tensor -v
```

- [ ] **2. exponent가 0이면 해당 경로의 기여가 0이므로 미분식 평가를 생략한다.**

```python
def backward() -> None:
    if exponent == 0:
        return
    self._accumulate(output.grad * exponent * self.data ** (exponent - 1))
```

기존 gradient를 0으로 덮어쓰지 않는다. 음수·분수 지수의 정의역 정책은 변경하지 않는다. Task 1에서 추가한 backward 등록 조건을 유지한다.

- [ ] **3. Tensor 테스트를 재실행해 새 경계값과 기존 제곱·공유 그래프 검사가 통과하는지 확인한다.**

**완료 기준:** 0제곱 경로에서 NaN·경고가 발생하지 않고 gradient 기여가 0이며, 다른 경로에서 누적된 gradient를 보존한다.

## Task 5: 수치 미분 예외 시 입력 복원

**Files:** `neural_engine/verification.py`, `tests/test_gradient_check.py`.

- [ ] **1. 양쪽 평가 단계의 예외에 대한 회귀 테스트를 추가한다.**

```python
def test_numerical_gradient_restores_input_on_evaluation_failure(self):
    for fail_on_call in (1, 2, 3, 4):
        with self.subTest(fail_on_call=fail_on_call):
            values = np.array([1.0, 2.0])
            original = values.copy()
            calls = 0

            def objective(array):
                nonlocal calls
                calls += 1
                if calls == fail_on_call:
                    raise RuntimeError("evaluation failed")
                return float((array * array).sum())

            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                numerical_gradient(objective, values)
            np.testing.assert_array_equal(values, original)
```

정상 반환 시 입력도 원래 값과 동일한지 기존 quadratic 테스트에 assertion을 추가한다.

- [ ] **2. `python3 -m unittest tests.test_gradient_check -v`로 변경 전 실패를 확인한다.**
- [ ] **3. 현재 원소의 perturbation 구간을 `try/finally`로 감싼다.**

```python
original = float(array[index])
try:
    array[index] = original + epsilon
    positive = float(function(array))
    array[index] = original - epsilon
    negative = float(function(array))
finally:
    array[index] = original
gradient[index] = (positive - negative) / (2.0 * epsilon)
```

호출자가 전달한 모델 파라미터 배열을 평가 함수가 함께 참조할 수 있으므로, 무조건 복사본으로 바꾸는 대신 현재 in-place perturbation 동작을 보존한다. 평가 함수가 입력의 다른 원소를 임의 변경하는 경우까지 복구하는 트랜잭션은 범위 밖이다. 원래 예외를 그대로 전파한다.

- [ ] **4. Gradient Check 테스트와 CLI를 재실행한다.**

```bash
python3 -m unittest tests.test_gradient_check -v
python3 scripts/gradient_check.py --log-file /tmp/vnte-quality-gradient-check.log
```

**완료 기준:** 모든 평가 단계에서 실패해도 함수가 변경한 원소가 복원되며, 원래 예외가 유지된다. 정상 수치 기울기와 기존 13개 검증의 정확도가 유지된다.

## Task 6: 통합 검증과 최종 검토

- [ ] **1. 전체 테스트를 한 번 실행한다.**

```bash
python3 -m unittest discover -s tests -v
```

기존 43개 및 새 회귀 테스트가 통과해야 한다. 기존 실패 재현 테스트를 단순히 삭제하거나 assertion을 약화하지 않는다.

- [ ] **2. 전체 MNIST 1 epoch를 실행해 추론 변경과 기본값 통일의 영향을 확인한다.**

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/train_mnist.py --epochs 1 --log-file /tmp/vnte-quality-mnist.log
```

전체 60,000/10,000 데이터와 기본 seed 42, hidden 256을 사용한다. 정확도 95% 이상, 로그의 유한한 loss·accuracy, 종료 코드 0을 확인한다. 진단 기준 `95.21%`와 차이가 나면 파라미터·배치 순서·추론 결과를 조사한다. 부동소수점 결과의 비트 단위 일치를 플랫폼 간 보장으로 추가하지 않는다.

- [ ] **3. 변경 범위와 결과를 검토한다.**

```bash
git diff --check
git diff --stat
git status --short
```

검토 항목:

- `predict_mnist`에 레이어 종류를 나열한 분기나 가중치 기반 별도 순전파가 남지 않는다.
- 비기록 결과에 부모 참조나 backward closure가 남지 않는다.
- 기본값 256이 공통 상수에서 나오고, 명시적 옵션은 유지된다.
- 실험 리포트가 관측 결과를 넘어 일반적인 우열·무결성을 단정하지 않는다.
- 0제곱 수정이 gradient 누적을 깨지 않는다.
- 수치 미분의 입력 복원과 예외 전달이 성공·실패 경로 모두에서 유지된다.
- 추적 중인 `logs/`, `figures/`, 데이터 캐시는 의도치 않게 변경되지 않는다.

- [ ] **4. 변경 내용·실행한 검증·남은 제한을 보고한다. 커밋·푸시는 별도 요청 범위에 따른다.**

## 계획 자체 검토 결과

- 사용자 항목 1 → Task 1, 항목 2 → Task 2, 항목 3 → Task 3, 항목 4 → Task 4·5로 대응한다.
- Task 1이 연산 결과 생성 경로에 영향을 주므로 Task 4는 해당 변경 이후 적용한다. 최종 검증은 모든 구현 이후 수행한다.
- 신규 런타임 의존성은 없다. 새 모듈은 그래프 기록 컨텍스트 하나로 제한한다.
- 이번 산출물은 이 계획 문서 한 개다. 구현·테스트 추가·실험 재실행은 시작하지 않은 상태로 대기한다.
