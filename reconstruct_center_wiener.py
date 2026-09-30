"""中心相位估计后，用50帧调制PSF做完整600×600 Wiener重建。"""

import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from config import (
    CENTER_BATCH, CENTER_EPOCHS, CENTER_FINAL_LR, CENTER_IMAGE_LR,
    CENTER_PHASE_LAYERS, CENTER_PHASE_LR, CENTER_WINDOWS,
    CENTER_ZERNIKE_MODES, DATA_DIR, DEVICE, FRAMES, OUTPUT_DIR, SEED,
    SLM_PHASE_SIGN, WIENER_REGULARIZATION,
)
from data import center_training_arrays, load_experiment, resize_real
from model import StaticNeuWS


def remove_piston_tip_tilt(phase: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(phase.shape, dtype=np.float64)
    matrix = np.column_stack((np.ones(phase.size), xx.ravel(), yy.ravel()))
    coefficients = np.linalg.lstsq(matrix, phase.ravel(), rcond=None)[0]
    plane = coefficients[0] + coefficients[1] * xx + coefficients[2] * yy
    return (phase - plane).astype(np.float32)


def resize_complex_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    amplitude = resize_real(np.abs(field), shape)
    unit = np.exp(1j * np.angle(field))
    phasor = resize_real(unit.real, shape) + 1j * resize_real(unit.imag, shape)
    phasor /= np.maximum(np.abs(phasor), 1e-8)
    resized = amplitude * phasor
    energy = float(np.mean(np.abs(resized) ** 2))
    return (resized / math.sqrt(max(energy, 1e-12))).astype(np.complex64)


def multiframe_wiener(measurements, slm_phase, system_field, regularization):
    frames, size, _ = measurements.shape
    fft_shape = (2 * size, 2 * size)
    half = size // 2
    numerator = np.zeros((fft_shape[0], fft_shape[1] // 2 + 1), np.complex128)
    denominator = np.zeros(numerator.shape, np.float64)

    for index in range(frames):
        pupil = system_field * np.exp(1j * SLM_PHASE_SIGN * slm_phase[index])
        psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil, norm="forward"))) ** 2
        psf /= max(float(psf.sum()), 1e-12)
        psf = psf[::-1, ::-1]

        transfer = np.fft.rfftn(psf, s=fft_shape)
        padded_measurement = np.zeros(fft_shape, np.float32)
        padded_measurement[half:half + size, half:half + size] = measurements[index]
        observed = np.fft.rfftn(padded_measurement, s=fft_shape)
        numerator += np.conj(transfer) * observed
        denominator += np.abs(transfer) ** 2

    scale = max(float(np.percentile(denominator, 95)), 1e-12)
    image = np.fft.irfftn(
        numerator / (denominator + regularization * scale), s=fft_shape).real
    image = np.clip(image[:size, :size], 0, None)
    return (image / max(float(np.percentile(image, 99.9)), 1e-8)).astype(np.float32)


torch.manual_seed(SEED)
np.random.seed(SEED)
experiment = load_experiment(DATA_DIR, FRAMES)

for center_size in CENTER_WINDOWS:
    center_measurements, center_phase = center_training_arrays(experiment, center_size)
    targets = torch.from_numpy(
        center_measurements / experiment["measurement_max"]
    ).float()
    slm_field = torch.exp(
        1j * SLM_PHASE_SIGN * torch.from_numpy(center_phase).float()
    ).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(slm_field, targets), batch_size=CENTER_BATCH, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    network = StaticNeuWS(
        center_size, CENTER_ZERNIKE_MODES, CENTER_PHASE_LAYERS).to(DEVICE)
    image_optimizer = torch.optim.Adam(network.object.parameters(), lr=CENTER_IMAGE_LR)
    phase_optimizer = torch.optim.Adam(network.aberration.parameters(), lr=CENTER_PHASE_LR)
    image_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        image_optimizer, CENTER_EPOCHS, eta_min=CENTER_FINAL_LR)
    phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        phase_optimizer, CENTER_EPOCHS, eta_min=CENTER_FINAL_LR)

    print(f"开始中心训练：window={center_size}, device={DEVICE}")
    history = []
    started = time.time()
    for epoch in trange(CENTER_EPOCHS):
        losses = []
        for batch_slm, batch_measurement in loader:
            batch_slm = batch_slm.to(DEVICE)
            batch_measurement = batch_measurement.to(DEVICE)
            image_optimizer.zero_grad(set_to_none=True)
            phase_optimizer.zero_grad(set_to_none=True)
            prediction, _, _, _, _ = network(batch_slm)
            loss = F.mse_loss(prediction, batch_measurement)
            loss.backward()
            image_optimizer.step()
            phase_optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        image_scheduler.step()
        phase_scheduler.step()

    network.eval()
    with torch.no_grad():
        _, raw_field, network_phase = network.estimates()
        raw_field = raw_field[0, 0].cpu().numpy()
        network_phase = network_phase[0, 0].cpu().numpy()

    signed_amplitude = np.real(raw_field * np.exp(-1j * network_phase))
    physical_phase = network_phase + np.where(signed_amplitude < 0, np.pi, 0.0)
    clean_phase = remove_piston_tip_tilt(physical_phase)
    center_field = np.abs(raw_field) * np.exp(1j * clean_phase)
    full_field = resize_complex_field(center_field, (600, 600))

    reconstruction = multiframe_wiener(
        experiment["measurements"], experiment["slm_phase"], full_field,
        WIENER_REGULARIZATION,
    )

    output = OUTPUT_DIR / f"center_{center_size:03d}_wiener"
    output.mkdir(parents=True, exist_ok=True)
    sio.savemat(output / "reconstruction.mat", {
        "image": reconstruction,
        "center_field": center_field,
        "center_phase": clean_phase,
        "full_field": full_field,
        "loss": np.asarray(history, np.float32),
    })
    torch.save(network.state_dict(), output / "center_model.pt")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    axes[0].imshow(center_measurements.mean(axis=0), cmap="gray")
    axes[0].set_title(f"中心观测 {center_size}×{center_size}")
    axes[1].imshow(clean_phase, cmap="twilight")
    axes[1].set_title("中心恢复相位")
    axes[2].imshow(np.angle(full_field), cmap="twilight")
    axes[2].set_title("插值后的600×600相位")
    axes[3].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("50帧全场Wiener重建")
    for axis in axes:
        axis.set(xlabel="x (px)", ylabel="y (px)")
    fig.savefig(output / "summary.png", dpi=180)
    plt.close(fig)

    (output / "summary.json").write_text(json.dumps({
        "center_size": center_size,
        "frames": FRAMES,
        "epochs": CENTER_EPOCHS,
        "regularization": WIENER_REGULARIZATION,
        "device": DEVICE,
        "elapsed_seconds": time.time() - started,
        "final_mse": history[-1],
    }, indent=2), encoding="utf-8")
    print(f"完成：{output}")
