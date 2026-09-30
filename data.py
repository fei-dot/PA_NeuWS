"""实验数据读取。这里不包含网络和重建算法。"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage


def load_mat(path: Path, variable: str) -> np.ndarray:
    return np.asarray(sio.loadmat(path)[variable]).squeeze()


def load_experiment(data_dir: Path, frames: int) -> dict:
    measurements = np.stack([
        load_mat(data_dir / f"SLM_raw{i}.mat", "imsdata")
        for i in range(1, frames + 1)
    ]).astype(np.float32)
    slm_phase = np.stack([
        load_mat(data_dir / f"SLM_sim{i}.mat", "proj_sim")
        for i in range(1, frames + 1)
    ]).astype(np.float32)

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    measurement_max = float(manifest.get("measurement_max", measurements.max()))

    truth_path = data_dir / "reference" / "clear_object.npy"
    ground_truth = np.load(truth_path).astype(np.float32) if truth_path.exists() else None

    print(f"读取实验数据：measurements={measurements.shape}, slm_phase={slm_phase.shape}")
    print(f"50帧共享强度归一化值：{measurement_max:.6g}")
    return {
        "measurements": measurements,
        "slm_phase": slm_phase,
        "measurement_max": measurement_max,
        "ground_truth": ground_truth,
    }


def resize_real(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    zoom = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    value = ndimage.zoom(array, zoom, order=3, mode="nearest", prefilter=True)
    return value[:shape[0], :shape[1]]


def resize_phase(phase: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    unit = np.exp(1j * phase.astype(np.float32))
    real = resize_real(unit.real, shape)
    imag = resize_real(unit.imag, shape)
    return np.angle(real + 1j * imag).astype(np.float32)


def center_training_arrays(experiment: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    measurements = experiment["measurements"]
    top = (measurements.shape[-2] - size) // 2
    left = (measurements.shape[-1] - size) // 2
    center_measurements = measurements[:, top:top + size, left:left + size]
    center_slm_phase = np.stack([
        resize_phase(phase, (size, size))
        for phase in experiment["slm_phase"]
    ])
    print(f"中心训练数据：measurement={center_measurements.shape}, phase={center_slm_phase.shape}")
    return center_measurements, center_slm_phase

