import gzip
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from neural_engine.data.mnist import (
    MNIST_FILES,
    batch_iterator,
    load_mnist,
    parse_idx,
)
from neural_engine.core.tensor import Tensor
from neural_engine.mnist_training import (
    build_mnist_model,
    predict_mnist,
    train_mnist,
)
from neural_engine.nn.activations import ReLU, Sigmoid, Softmax
from neural_engine.nn.layers import Linear
from neural_engine.nn.module import Module, Sequential
from neural_engine.cli.train_mnist import main as train_mnist_main


def image_idx(values: np.ndarray) -> bytes:
    count, rows, columns = values.shape
    return struct.pack(">IIII", 2051, count, rows, columns) + values.tobytes()


def label_idx(values: np.ndarray) -> bytes:
    return struct.pack(">II", 2049, values.size) + values.tobytes()


class MnistDataTests(unittest.TestCase):
    def test_parse_idx_images(self):
        payload = struct.pack(">IIII", 2051, 2, 2, 2) + bytes(range(8))

        result = parse_idx(payload)

        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result.reshape(-1), np.arange(8))

    def test_parse_idx_labels(self):
        payload = struct.pack(">II", 2049, 3) + bytes([7, 1, 4])

        result = parse_idx(payload)

        np.testing.assert_array_equal(result, [7, 1, 4])

    def test_parse_idx_rejects_bad_magic_and_payload_size(self):
        with self.assertRaisesRegex(ValueError, "magic"):
            parse_idx(struct.pack(">II", 999, 0))
        with self.assertRaisesRegex(ValueError, "payload size"):
            parse_idx(struct.pack(">II", 2049, 2) + bytes([1]))

    def test_load_mnist_parses_gzip_cache_and_normalizes_images(self):
        train_images = np.array([[[0, 255], [128, 64]]], dtype=np.uint8)
        train_labels = np.array([3], dtype=np.uint8)
        test_images = np.array([[[255, 0], [32, 16]]], dtype=np.uint8)
        test_labels = np.array([7], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "train_images": image_idx(train_images),
                "train_labels": label_idx(train_labels),
                "test_images": image_idx(test_images),
                "test_labels": label_idx(test_labels),
            }
            for key, payload in fixtures.items():
                (root / MNIST_FILES[key]).write_bytes(gzip.compress(payload))

            loaded = load_mnist(root, download=False)

        x_train, y_train, x_test, y_test = loaded
        self.assertEqual(x_train.shape, (1, 4))
        self.assertEqual(x_train.dtype, np.float32)
        np.testing.assert_allclose(x_train[0], [0.0, 1.0, 128 / 255, 64 / 255])
        np.testing.assert_array_equal(y_train, [3])
        np.testing.assert_array_equal(y_test, [7])
        self.assertEqual(x_test.shape, (1, 4))

    def test_batch_iterator_is_complete_and_seeded(self):
        inputs = np.arange(20).reshape(10, 2)
        targets = np.arange(10)

        first = list(
            batch_iterator(inputs, targets, 3, np.random.default_rng(42), shuffle=True)
        )
        second = list(
            batch_iterator(inputs, targets, 3, np.random.default_rng(42), shuffle=True)
        )

        self.assertEqual([len(batch[0]) for batch in first], [3, 3, 3, 1])
        np.testing.assert_array_equal(
            np.concatenate([batch[1] for batch in first]),
            np.concatenate([batch[1] for batch in second]),
        )
        self.assertEqual(set(np.concatenate([batch[1] for batch in first])), set(range(10)))

    def test_predict_mnist_matches_fixed_model_predictions(self):
        first = Linear(2, 2, initialization="zero")
        second = Linear(2, 2, initialization="zero")
        first.weight.data[:] = np.eye(2)
        second.weight.data[:] = np.eye(2)
        model = Sequential(first, ReLU(), second)

        predictions = predict_mnist(
            model, np.array([[1.0, -2.0], [-1.0, 3.0]]), batch_size=1
        )

        np.testing.assert_array_equal(predictions, [0, 1])

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

    def test_predict_matches_sequential_model_forward(self):
        model = Sequential(
            Linear(2, 3, initialization="zero"),
            ReLU(),
            Linear(3, 2, initialization="zero"),
            Softmax(),
        )
        model.layers[0].weight.data[:] = np.array(
            [[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]
        )
        model.layers[2].weight.data[:] = np.array(
            [[1.0, -1.0], [0.5, 0.25], [-0.5, 1.0]]
        )
        inputs = np.array([[1.0, 2.0], [-2.0, 1.0]])

        expected = np.argmax(model(Tensor(inputs)).data, axis=1)

        np.testing.assert_array_equal(predict_mnist(model, inputs), expected)

    def test_predict_rejects_invalid_batch_size_and_output_shape(self):
        model = Sequential(Linear(2, 2, initialization="zero"))
        inputs = np.ones((2, 2))

        with self.assertRaisesRegex(ValueError, "batch_size"):
            predict_mnist(model, inputs, batch_size=0)

        class BadModel(Module):
            def forward(self, inputs):
                return Tensor(np.ones(inputs.shape[0]))

        with self.assertRaisesRegex(ValueError, "shape"):
            predict_mnist(BadModel(), inputs)

        np.testing.assert_array_equal(
            predict_mnist(model, np.empty((0, 2))), np.empty(0, dtype=np.int64)
        )

    def test_predict_preserves_parameter_data_and_gradients(self):
        model = Sequential(Linear(2, 2, initialization="zero"))
        model.layers[0].weight.data[:] = np.eye(2)
        model.layers[0].bias.data[:] = [1.0, -1.0]
        model(Tensor([[1.0, 2.0]])).sum().backward()
        data_before = [parameter.data.copy() for parameter in model.parameters()]
        grad_before = [parameter.grad.copy() for parameter in model.parameters()]

        predict_mnist(model, np.array([[2.0, 1.0]]))

        for parameter, expected_data, expected_grad in zip(
            model.parameters(), data_before, grad_before
        ):
            np.testing.assert_array_equal(parameter.data, expected_data)
            np.testing.assert_array_equal(parameter.grad, expected_grad)

    def test_train_mnist_defaults_to_256_hidden_features(self):
        inputs = np.eye(2, dtype=np.float32)
        targets = np.array([0, 1])

        with patch(
            "neural_engine.mnist_training.build_mnist_model",
            wraps=build_mnist_model,
        ) as builder:
            train_mnist(
                inputs,
                targets,
                inputs,
                targets,
                epochs=1,
                batch_size=2,
                class_count=2,
            )

        self.assertEqual(builder.call_args.kwargs["hidden_features"], 256)
        self.assertEqual(build_mnist_model().layers[0].out_features, 256)

    def test_tiny_mnist_training_records_loss_and_accuracy(self):
        inputs = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]],
            dtype=np.float32,
        )
        targets = np.array([0, 1, 0, 1])

        history = train_mnist(
            inputs,
            targets,
            inputs,
            targets,
            epochs=30,
            batch_size=4,
            learning_rate=0.05,
            seed=42,
            hidden_features=4,
            class_count=2,
        )

        self.assertEqual(len(history), 30)
        self.assertLess(history[-1].loss, history[0].loss)
        self.assertGreaterEqual(history[-1].accuracy, 0.75)

    def test_training_rejects_empty_or_mismatched_splits(self):
        inputs = np.ones((2, 4), dtype=np.float32)
        targets = np.array([0, 1])

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            train_mnist(
                inputs[:0],
                targets[:0],
                inputs,
                targets,
                epochs=1,
                class_count=2,
            )
        with self.assertRaisesRegex(ValueError, "same length"):
            train_mnist(
                inputs,
                targets,
                inputs,
                targets[:1],
                epochs=1,
                class_count=2,
            )

    def test_training_cli_uses_cached_idx_files_and_writes_log(self):
        images = np.array([[[0, 255], [0, 255]], [[255, 0], [255, 0]]], dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "train_images": image_idx(images),
                "train_labels": label_idx(labels),
                "test_images": image_idx(images),
                "test_labels": label_idx(labels),
            }
            for key, payload in fixtures.items():
                (root / MNIST_FILES[key]).write_bytes(gzip.compress(payload))
            log_file = root / "mnist.log"

            status = train_mnist_main(
                [
                    "--data-dir",
                    str(root),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--hidden-features",
                    "4",
                    "--train-limit",
                    "2",
                    "--test-limit",
                    "2",
                    "--log-file",
                    str(log_file),
                ]
            )

            self.assertEqual(status, 0)
            self.assertIn("epoch,loss,accuracy,seed", log_file.read_text())

    def test_training_cli_defaults_to_256_and_honors_explicit_hidden_features(self):
        images = np.array([[[0, 255], [0, 255]], [[255, 0], [255, 0]]], dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "train_images": image_idx(images),
                "train_labels": label_idx(labels),
                "test_images": image_idx(images),
                "test_labels": label_idx(labels),
            }
            for key, payload in fixtures.items():
                (root / MNIST_FILES[key]).write_bytes(gzip.compress(payload))

            with patch(
                "neural_engine.cli.train_mnist.train_mnist", wraps=train_mnist
            ) as trainer:
                train_mnist_main(
                    [
                        "--data-dir",
                        str(root),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--train-limit",
                        "2",
                        "--test-limit",
                        "2",
                        "--log-file",
                        str(root / "default.log"),
                    ]
                )
                self.assertEqual(trainer.call_args.kwargs["hidden_features"], 256)

                train_mnist_main(
                    [
                        "--data-dir",
                        str(root),
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--hidden-features",
                        "4",
                        "--train-limit",
                        "2",
                        "--test-limit",
                        "2",
                        "--log-file",
                        str(root / "explicit.log"),
                    ]
                )
                self.assertEqual(trainer.call_args.kwargs["hidden_features"], 4)


if __name__ == "__main__":
    unittest.main()
