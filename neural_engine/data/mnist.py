from __future__ import annotations

import gzip
import shutil
import struct
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
)


def parse_idx(payload: bytes) -> np.ndarray:
    if len(payload) < 4:
        raise ValueError("IDX header is shorter than 4 bytes")
    magic = struct.unpack(">I", payload[:4])[0]
    if magic not in (2049, 2051):
        raise ValueError(f"unsupported IDX magic number: {magic}")
    dimensions = magic & 0xFF
    header_size = 4 + 4 * dimensions
    if len(payload) < header_size:
        raise ValueError("IDX dimension header is truncated")
    shape = struct.unpack(f">{dimensions}I", payload[4:header_size])
    expected_size = int(np.prod(shape))
    actual_size = len(payload) - header_size
    if actual_size != expected_size:
        raise ValueError(
            f"IDX payload size mismatch: expected {expected_size}, got {actual_size}"
        )
    return np.frombuffer(payload, dtype=np.uint8, offset=header_size).reshape(shape).copy()


def ensure_mnist(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in MNIST_FILES.items():
        destination = root / filename
        paths[key] = destination
        if destination.exists():
            continue

        temporary = destination.with_suffix(destination.suffix + ".download")
        errors: list[str] = []
        for base_url in MNIST_BASE_URLS:
            url = f"{base_url}/{filename}"
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle)
                temporary.replace(destination)
                break
            except (OSError, urllib.error.URLError) as error:
                errors.append(f"{url}: {error}")
                temporary.unlink(missing_ok=True)
        else:
            details = "\n".join(errors)
            raise RuntimeError(f"failed to download {filename}:\n{details}")
    return paths


def load_mnist(
    data_dir: str | Path = "data",
    *,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    if download:
        paths = ensure_mnist(root)
    else:
        paths = {key: root / filename for key, filename in MNIST_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"MNIST cache files are missing: {', '.join(missing)}")

    arrays = {}
    for key, path in paths.items():
        try:
            with gzip.open(path, "rb") as handle:
                arrays[key] = parse_idx(handle.read())
        except (gzip.BadGzipFile, EOFError) as error:
            raise ValueError(f"invalid gzip file {path}: {error}") from error

    train_images = arrays["train_images"]
    train_labels = arrays["train_labels"]
    test_images = arrays["test_images"]
    test_labels = arrays["test_labels"]
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not match")

    x_train = train_images.reshape(len(train_images), -1).astype(np.float32) / 255.0
    x_test = test_images.reshape(len(test_images), -1).astype(np.float32) / 255.0
    return x_train, train_labels.astype(np.int64), x_test, test_labels.astype(np.int64)


def batch_iterator(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    *,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(inputs))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        yield inputs[selected], targets[selected]
