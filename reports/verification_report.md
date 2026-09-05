# 검증 리포트

## 검증 목적

AutoGrad가 계산한 analytic gradient를 중앙 차분 numerical gradient와 비교하여 Tensor 연산과 필수 레이어의 역전파 구현이 정확한지 학습 전에 독립적으로 검증했다.

## AutoGrad 구조

각 Tensor는 `data`, `grad`, 부모 Tensor 집합, 연산 이름과 로컬 `_backward()`를 저장한다. 순전파 과정에서 그래프가 동적으로 연결된다. `backward()`는 출력 Tensor에서 DFS로 위상 순서를 만들고, 출력 기울기를 설정한 뒤 역순으로 각 `_backward()`를 실행한다.

로컬 기울기에는 출력 측 기울기를 곱해 Chain Rule을 적용하고, 여러 경로에서 같은 Tensor로 도착한 기울기는 합산한다. leaf gradient는 의도적으로 누적되므로 각 학습 step의 역전파 전에 `optimizer.zero_grad()`가 필요하다.

## Broadcasting 처리

NumPy broadcasting이 적용된 연산의 역전파에서는 gradient shape를 원래 피연산자 shape로 복원한다. `sum_to_shape()`가 순전파에서 추가된 선행 축을 먼저 합산하고, 크기 1이었던 축을 유지한 채 합산한다. Bias가 `(3,)`에서 `(2, 3)`으로 확장된 검증 사례에서는 역전파 시 batch 축 두 행의 기울기가 `(3,)`으로 합쳐진다.

## 수치 미분 방법

epsilon은 원문 기준 `1e-5`를 사용했다.

```text
numerical gradient = (f(x + ε) - f(x - ε)) / (2ε)
relative error = max(|analytic - numerical| / max(1e-12, |analytic| + |numerical|))
```

수치 미분은 원소마다 최소 두 번의 순전파가 필요해 계산 비용이 크고 반올림 오차가 있다. 따라서 실제 학습이 아니라 작은 고정 입력에서 역전파 수식의 무결성을 확인하는 데 사용했다. ReLU는 0에서 미분 불가능하므로 `[-1.2, -0.3, 0.4, 1.5]`를 사용했다.

## 실행 방법

```bash
neural-engine verify gradients
```

## 실행 결과

실행일: 2026-08-31, seed: `42`, 기준: `1e-7`

| 대상 | 최대 상대오차 | 판정 |
|---|---:|---|
| add + broadcasting | `1.438e-11` | PASS |
| multiply | `2.969e-12` | PASS |
| divide | `9.707e-11` | PASS |
| matmul | `2.645e-11` | PASS |
| sum / mean | `3.276e-12` | PASS |
| Linear input | `1.646e-11` | PASS |
| Linear weight | `8.377e-11` | PASS |
| Linear bias | `4.126e-12` | PASS |
| ReLU | `8.210e-12` | PASS |
| Sigmoid | `3.071e-11` | PASS |
| Softmax | `8.684e-11` | PASS |
| Binary Cross Entropy | `1.049e-10` | PASS |
| Cross Entropy | `1.694e-11` | PASS |

전체 최대 상대오차는 Binary Cross Entropy의 `1.049e-10`이며 허용 기준 `1e-7`보다 작다. 저장된 원본 출력은 `logs/gradient_check.log`에 있다.

## 결론

주요 Tensor 연산, broadcasting 축소, Linear의 입력·weight·bias 및 모든 필수 활성화 레이어가 기준을 통과했다. 따라서 XOR와 MNIST 학습 결과를 역전파 수식 오류가 아닌 최적화·초기화 조건의 결과로 해석할 수 있다.
