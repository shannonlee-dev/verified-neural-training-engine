# Verified Neural Training Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** NumPy 동적 AutoGrad 엔진을 구현하고 XOR·MNIST 학습과 초기화 실험으로 정확성과 재현성을 입증한다.

**Architecture:** `Tensor`가 동적 계산 그래프와 역전파를 담당하고, `Module` 기반 레이어가 이를 조합한다. 옵티마이저·데이터 로더·실험 스크립트는 작은 공개 API만 의존하며, 표준 `unittest`와 저장 로그로 단위·통합 동작을 검증한다.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, Python 표준 `unittest`, `urllib`, `gzip`, `struct`

**Spec:** `docs/superpowers/specs/2026-08-31-neural-training-engine-design.md`

## Global Constraints

- 모든 학습·자동미분 계산은 NumPy로 구현하며 외부 ML/DL 프레임워크를 사용하지 않는다.
- Gradient Check 중앙 차분 epsilon은 `1e-5`, 상대오차 완료 기준은 `<= 1e-7`이다.
- ReLU Gradient Check 입력은 0을 피한다.
- 초기화는 Zero, Random, He, Xavier를 모두 제공한다.
- MNIST는 IDX gzip을 직접 파싱하고 `data/` 캐시를 사용한다.
- 테스트 러너는 Python 표준 `unittest`이며 requirements는 NumPy와 Matplotlib만 포함한다.
- 기본 seed는 `42`이고 모든 CLI에서 `--seed`로 변경할 수 있다.
- 학습 단계는 `zero_grad → forward → loss → backward → step` 순서를 지킨다.

---

### Task 1: 프로젝트 기반과 Tensor AutoGrad

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `neural_engine/__init__.py`
- Create: `neural_engine/config.py`
- Create: `neural_engine/core/__init__.py`
- Create: `neural_engine/core/utils.py`
- Create: `neural_engine/core/tensor.py`
- Create: `tests/__init__.py`
- Create: `tests/test_tensor.py`

**Interfaces:**
- Produces: `DEFAULT_SEED: int`, `sum_to_shape(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray`, `Tensor(data, requires_grad=False)`, Tensor arithmetic, `matmul`, `sum`, `mean`, `exp`, `log`, `reshape`, indexing and `backward(gradient=None)`.

- [ ] **Step 1: Write failing tests for graph traversal, gradient accumulation and broadcasting**

```python
class TensorTests(unittest.TestCase):
    def test_backward_accumulates_through_shared_graph(self):
        x = Tensor([2.0], requires_grad=True)
        y = x * x + x
        y.backward()
        np.testing.assert_allclose(x.grad, [5.0])
        y.backward()
        np.testing.assert_allclose(x.grad, [10.0])

    def test_broadcast_gradient_returns_original_shape(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        bias = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        (x + bias).sum().backward()
        np.testing.assert_allclose(bias.grad, [2.0, 2.0, 2.0])
```

- [ ] **Step 2: Run the focused tests and confirm missing imports fail**

Run: `python3 -m unittest tests.test_tensor -v`

Expected: FAIL because `neural_engine.core.tensor` does not exist.

- [ ] **Step 3: Implement Tensor and shape reduction**

Use a `set` of parent tensors, DFS topological ordering and reverse traversal. Before traversal, clear gradients only on non-leaf graph nodes so a repeated call recomputes the same local derivatives while leaf gradients remain cumulative. Each local `_backward` adds into parent gradients with `sum_to_shape`; scalar backward defaults to `np.ones_like(self.data)` and non-scalar outputs require an explicit matching gradient.

```python
def sum_to_shape(gradient, shape):
    result = np.asarray(gradient, dtype=np.float64)
    while result.ndim > len(shape):
        result = result.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    return result.reshape(shape)
```

- [ ] **Step 4: Add numeric cases for divide, matmul, reductions, exp, log and indexing**

Add these explicit expectations before implementing the corresponding operators:

```python
def test_required_tensor_operations_propagate_gradients(self):
    left = Tensor([[1.0, 2.0]], requires_grad=True)
    right = Tensor([[3.0], [4.0]], requires_grad=True)
    ((left @ right) / 2.0).mean().backward()
    np.testing.assert_allclose(left.grad, [[1.5, 2.0]])
    np.testing.assert_allclose(right.grad, [[0.5], [1.0]])

def test_exp_log_and_indexing_propagate(self):
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.exp().log()[[0, 2]].sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0])

def test_non_scalar_backward_requires_gradient(self):
    with self.assertRaises(ValueError):
        Tensor([1.0, 2.0], requires_grad=True).backward()
```

- [ ] **Step 5: Run Task 1 tests**

Run: `python3 -m unittest tests.test_tensor -v`

Expected: all Tensor tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add .gitignore requirements.txt neural_engine tests/test_tensor.py tests/__init__.py
git commit -m "feat: add tensor autograd engine"
```

---

### Task 2: Module, 레이어, 활성화, 손실과 초기화

**Files:**
- Create: `neural_engine/nn/__init__.py`
- Create: `neural_engine/nn/module.py`
- Create: `neural_engine/nn/initialization.py`
- Create: `neural_engine/nn/layers.py`
- Create: `neural_engine/nn/activations.py`
- Create: `neural_engine/nn/losses.py`
- Create: `tests/test_nn.py`

**Interfaces:**
- Consumes: `Tensor`, `DEFAULT_SEED`.
- Produces: `Module.parameters()`, `Sequential`, `Linear(in_features, out_features, initialization, rng)`, `ReLU`, `Sigmoid`, `Softmax(axis=-1)`, `binary_cross_entropy`, `cross_entropy`, `initialize_weights`.

- [ ] **Step 1: Write failing initialization and module tests**

```python
def test_initializers_have_expected_values_and_scales(self):
    rng = np.random.default_rng(42)
    self.assertTrue(np.all(initialize_weights(4, 3, "zero", rng) == 0))
    he = initialize_weights(4000, 3, "he", rng)
    self.assertAlmostEqual(float(he.std()), np.sqrt(2 / 4000), delta=0.002)
    xavier = initialize_weights(4000, 2000, "xavier", rng)
    self.assertAlmostEqual(float(xavier.std()), np.sqrt(2 / 6000), delta=0.002)

def test_sequential_collects_linear_parameters(self):
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
    self.assertEqual(len(model.parameters()), 4)
```

- [ ] **Step 2: Verify Task 2 tests fail**

Run: `python3 -m unittest tests.test_nn -v`

Expected: FAIL because the NN modules do not exist.

- [ ] **Step 3: Implement initialization, Module, Sequential and Linear**

Use the four exact scales from the spec. `Module.__call__` delegates to `forward`; `Sequential.forward` applies layers in order; `Linear.forward(x)` returns `x @ weight + bias`.

- [ ] **Step 4: Write activation and loss behavior tests**

```python
def test_softmax_rows_are_probabilities(self):
    probabilities = Softmax()(Tensor([[1000.0, 1001.0, 1002.0]])).data
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])
    self.assertTrue(np.isfinite(probabilities).all())

def test_cross_entropy_is_stable_and_backpropagates(self):
    logits = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)
    loss = cross_entropy(logits, np.array([2]))
    loss.backward()
    self.assertTrue(np.isfinite(loss.data).all())
    np.testing.assert_allclose(logits.grad.sum(axis=1), [0.0], atol=1e-12)
```

- [ ] **Step 5: Implement ReLU, Sigmoid, stable Softmax and fused losses**

Softmax subtracts row maxima and its local backward computes `p * (g - sum(g*p))`. Cross Entropy computes log-sum-exp from logits and creates a scalar Tensor whose backward adds `(probabilities - one_hot) / batch_size` to logits. Binary Cross Entropy clips only for forward stability and adds its exact derivative to the probability Tensor.

- [ ] **Step 6: Run Tensor and NN tests**

Run: `python3 -m unittest tests.test_tensor tests.test_nn -v`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add neural_engine/nn tests/test_nn.py
git commit -m "feat: add neural network layers and losses"
```

---

### Task 3: SGD와 Adam

**Files:**
- Create: `neural_engine/optim/__init__.py`
- Create: `neural_engine/optim/optimizer.py`
- Create: `neural_engine/optim/sgd.py`
- Create: `neural_engine/optim/adam.py`
- Create: `tests/test_optim.py`

**Interfaces:**
- Consumes: iterable of trainable `Tensor` parameters.
- Produces: `Optimizer.zero_grad()`, `SGD(parameters, lr).step()`, `Adam(parameters, lr, betas, eps).step()`.

- [ ] **Step 1: Write failing optimizer tests**

```python
def test_sgd_updates_parameter_and_zeroes_gradient(self):
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([0.5])
    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()
    np.testing.assert_allclose(parameter.data, [1.95])
    optimizer.zero_grad()
    np.testing.assert_allclose(parameter.grad, [0.0])

def test_first_adam_step_matches_bias_corrected_formula(self):
    parameter = Tensor([2.0], requires_grad=True)
    parameter.grad = np.array([0.5])
    Adam([parameter], lr=0.1).step()
    np.testing.assert_allclose(parameter.data, [1.9], rtol=1e-7)
```

- [ ] **Step 2: Verify optimizer tests fail**

Run: `python3 -m unittest tests.test_optim -v`

Expected: FAIL because optimizer modules do not exist.

- [ ] **Step 3: Implement Optimizer, SGD and Adam**

Validate positive learning rates, keep Adam `m`/`v` arrays aligned with parameter order, increment one global step per `step()`, and apply both bias corrections before updating data in-place.

- [ ] **Step 4: Run all unit tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: all current tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add neural_engine/optim tests/test_optim.py
git commit -m "feat: add SGD and Adam optimizers"
```

---

### Task 4: Gradient Checking 시스템

**Files:**
- Create: `neural_engine/verification.py`
- Create: `scripts/__init__.py`
- Create: `scripts/gradient_check.py`
- Create: `tests/test_gradient_check.py`
- Create: `logs/.gitkeep`

**Interfaces:**
- Produces: `numerical_gradient(function, array, epsilon=1e-5)`, `relative_error(analytic, numerical)`, `run_gradient_checks() -> list[CheckResult]` and CLI exit status.

- [ ] **Step 1: Write failing numerical-gradient tests**

```python
def test_numerical_gradient_matches_quadratic(self):
    values = np.array([-2.0, 3.0])
    gradient = numerical_gradient(lambda x: float((x * x).sum()), values)
    np.testing.assert_allclose(gradient, 2 * values, rtol=1e-9, atol=1e-9)

def test_required_checks_meet_threshold(self):
    results = run_gradient_checks()
    required = {"add_broadcast", "multiply", "divide", "matmul", "sum_mean", "Linear.input", "Linear.weight", "Linear.bias", "ReLU", "Sigmoid", "Softmax"}
    self.assertTrue(required.issubset({result.name for result in results}))
    self.assertLessEqual(max(result.relative_error for result in results), 1e-7)
```

- [ ] **Step 2: Verify Gradient Check tests fail**

Run: `python3 -m unittest tests.test_gradient_check -v`

Expected: FAIL because verification module does not exist.

- [ ] **Step 3: Implement central differences and required cases**

Perturb one array element at a time and restore it after each pair. Use `max(abs(a-n) / max(1e-12, abs(a)+abs(n)))`. ReLU uses exactly `[-1.2, -0.3, 0.4, 1.5]` and no zero entry.

- [ ] **Step 4: Implement CLI logging and nonzero failure exit**

Format each row as `[PASS] <name>: relative_error=<scientific>` and finish with maximum error and threshold. Create the log parent directory, write the same text to `logs/gradient_check.log`, and return status 1 if any result exceeds `1e-7`.

- [ ] **Step 5: Run Gradient Check and full tests**

Run: `python3 scripts/gradient_check.py`

Expected: every line PASS and maximum relative error `<= 1e-7`.

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add neural_engine/verification.py scripts logs/.gitkeep tests/test_gradient_check.py
git commit -m "feat: add comprehensive gradient checking"
```

---

### Task 5: XOR 학습과 초기화 비교

**Files:**
- Create: `neural_engine/experiments.py`
- Create: `scripts/train_xor.py`
- Create: `scripts/compare_initialization.py`
- Create: `tests/test_xor.py`
- Create: `figures/.gitkeep`

**Interfaces:**
- Produces: `build_xor_model(initialization, seed)`, `train_xor(initialization, epochs, seed) -> list[EpochMetrics]`, CLI log output and comparison CSV/PNG.

- [ ] **Step 1: Write failing deterministic XOR tests**

```python
def test_he_initialization_solves_xor_within_100_epochs(self):
    history = train_xor("he", epochs=100, seed=42)
    self.assertTrue(history[-1].loss < 0.1 or history[-1].accuracy >= 0.95)

def test_zero_initialization_stays_unsolved_for_50_epochs(self):
    history = train_xor("zero", epochs=50, seed=42)
    self.assertLess(history[-1].accuracy, 0.95)
    self.assertGreater(history[-1].loss, 0.1)
```

- [ ] **Step 2: Verify XOR tests fail**

Run: `python3 -m unittest tests.test_xor -v`

Expected: FAIL because experiment functions do not exist.

- [ ] **Step 3: Implement shared XOR training**

Use a `2 → 8 → 1` ReLU/Sigmoid network, full-batch Adam, the fixed four XOR samples and one initialization argument. Record epoch, loss, thresholded accuracy, initialization and seed after every update. Tune only learning rate and hidden width if the deterministic test misses its declared criterion.

- [ ] **Step 4: Implement XOR and comparison CLIs**

`train_xor.py` accepts `--initialization`, `--epochs`, `--seed`, `--log-file`. `compare_initialization.py` runs zero/random/he at the same seed, saves columns `epoch,loss,accuracy,initialization,seed` to `logs/initialization_comparison.csv`, and uses the non-interactive Matplotlib `Agg` backend for `figures/initialization_loss.png`.

- [ ] **Step 5: Run required XOR evidence commands**

Run: `python3 scripts/train_xor.py --initialization he --epochs 100 --log-file logs/xor_he.log`

Expected: final loss `< 0.1` or accuracy `>= 95%`.

Run: `python3 scripts/train_xor.py --initialization zero --epochs 50 --log-file logs/xor_zero.log`

Expected: final loss `> 0.1` and accuracy `< 95%`.

Run: `python3 scripts/compare_initialization.py`

Expected: CSV and nonempty PNG are created.

- [ ] **Step 6: Commit Task 5**

```bash
git add neural_engine/experiments.py scripts/train_xor.py scripts/compare_initialization.py tests/test_xor.py figures/.gitkeep
git commit -m "feat: add XOR initialization experiments"
```

---

### Task 6: MNIST IDX 로더와 1 epoch 학습

**Files:**
- Create: `neural_engine/data/__init__.py`
- Create: `neural_engine/data/mnist.py`
- Create: `scripts/train_mnist.py`
- Create: `tests/test_mnist.py`

**Interfaces:**
- Produces: `parse_idx(payload: bytes) -> np.ndarray`, `ensure_mnist(data_dir)`, `load_mnist(data_dir)`, `batch_iterator`, MNIST training CLI.

- [ ] **Step 1: Write failing IDX parser tests**

```python
def test_parse_idx_images(self):
    payload = struct.pack(">IIII", 2051, 2, 2, 2) + bytes(range(8))
    result = parse_idx(payload)
    self.assertEqual(result.shape, (2, 2, 2))
    self.assertEqual(result.dtype, np.uint8)

def test_parse_idx_rejects_bad_magic(self):
    with self.assertRaisesRegex(ValueError, "magic"):
        parse_idx(struct.pack(">II", 999, 0))
```

- [ ] **Step 2: Verify MNIST loader tests fail**

Run: `python3 -m unittest tests.test_mnist -v`

Expected: FAIL because MNIST module does not exist.

- [ ] **Step 3: Implement direct IDX parsing and caching**

Map the four canonical gzip filenames to their download URLs. Download to a temporary sibling file and rename only after success. Parse gzip content directly, accept magic 2051 images and 2049 labels, validate declared payload size exactly and raise descriptive `ValueError` for malformed data.

- [ ] **Step 4: Add deterministic mini-batch and preprocessing tests**

Assert float32 `[0,1]` flattened images, integer labels, no missing samples, and identical batch order for identical seeds.

- [ ] **Step 5: Implement MNIST CLI**

Use `784 → 128 → 10`, ReLU, Cross Entropy, Adam, batch size 128 and one default epoch. Provide `--data-dir`, `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--train-limit`, `--test-limit` and `--log-file`. Evaluate test accuracy without building a gradient graph by reading `.data`; exit nonzero when the full default run finishes below 80%.

- [ ] **Step 6: Run offline unit tests and online training**

Run: `python3 -m unittest tests.test_mnist -v`

Expected: all parser and batching tests pass without network.

Run: `python3 scripts/train_mnist.py --epochs 1`

Expected: first execution downloads four gzip files, later execution reuses them, and test accuracy is `>= 80%`.

- [ ] **Step 7: Commit Task 6**

```bash
git add neural_engine/data scripts/train_mnist.py tests/test_mnist.py
git commit -m "feat: add direct MNIST training pipeline"
```

---

### Task 7: README, 리포트, 최종 산출물과 검증

**Files:**
- Create: `README.md`
- Create: `reports/verification_report.md`
- Create: `reports/experiment_report.md`
- Modify: `logs/gradient_check.log`
- Create: `logs/xor_he.log`
- Create: `logs/xor_zero.log`
- Create: `logs/mnist.log`
- Create: `logs/initialization_comparison.csv`
- Create: `figures/initialization_loss.png`

**Interfaces:**
- Consumes: all public APIs, scripts and actual experiment output.
- Produces: reproducible repository documentation and checked-in evidence.

- [ ] **Step 1: Regenerate every required artifact from a clean command sequence**

```bash
python3 scripts/gradient_check.py
python3 scripts/train_xor.py --initialization he --epochs 100 --log-file logs/xor_he.log
python3 scripts/train_xor.py --initialization zero --epochs 50 --log-file logs/xor_zero.log
python3 scripts/compare_initialization.py
python3 scripts/train_mnist.py --epochs 1 --log-file logs/mnist.log
```

- [ ] **Step 2: Write verification report from recorded values**

Include calculation graph order, Chain Rule, broadcasting reduction, central difference, relative-error formula, learning separation rationale, a table of every Gradient Check result and the actual maximum error.

- [ ] **Step 3: Write experiment report from recorded values**

Include XOR He success, Zero 50-epoch failure with both loss and accuracy, Random/He/Zero comparison, symmetry analysis, ReLU/He and Sigmoid/Xavier relationships, Adam as Momentum plus RMSProp families, and actual MNIST accuracy.

- [ ] **Step 4: Write README and checklist**

Open with overview, features, install, quick verification and training commands. Add architecture, Tensor/backward example, exact `zero_grad → forward → loss → backward → step` example, cumulative gradient policy, MNIST first-download/cache behavior, seed controls, actual result table, project tree and every mission checklist item.

- [ ] **Step 5: Run complete verification**

Run: `python3 -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `python3 scripts/gradient_check.py`

Expected: maximum relative error `<= 1e-7`.

Check: XOR logs meet their opposite success/failure criteria, MNIST log reports `>= 80%`, CSV has all five required columns, and PNG is nonempty.

- [ ] **Step 6: Inspect repository diff and commit final documentation/artifacts**

```bash
git diff --check
git add README.md reports logs figures requirements.txt .gitignore
git commit -m "docs: add verified training results"
```
