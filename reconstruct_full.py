"""完整600×600联合重建。配置→读取→训练→保存，按顺序直接运行。"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from config import (
    DATA_DIR, DEVICE, FRAMES, FULL_BATCH, FULL_EPOCHS, FULL_FINAL_LR,
    FULL_IMAGE_LR, FULL_PHASE_LAYERS, FULL_PHASE_LR, FULL_ZERNIKE_MODES,
    OUTPUT_DIR, SEED, SLM_PHASE_SIGN,
)
from data import load_experiment
from model import StaticNeuWS


torch.manual_seed(SEED)
np.random.seed(SEED)

experiment = load_experiment(DATA_DIR, FRAMES)
measurements = torch.from_numpy(
    experiment["measurements"] / experiment["measurement_max"]
).float()
slm_field = torch.exp(
    1j * SLM_PHASE_SIGN * torch.from_numpy(experiment["slm_phase"]).float()
).unsqueeze(1)

dataset = TensorDataset(slm_field, measurements)
loader = DataLoader(dataset, batch_size=FULL_BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))

size = measurements.shape[-1]
network = StaticNeuWS(size, FULL_ZERNIKE_MODES, FULL_PHASE_LAYERS).to(DEVICE)
image_optimizer = torch.optim.Adam(network.object.parameters(), lr=FULL_IMAGE_LR)
phase_optimizer = torch.optim.Adam(network.aberration.parameters(), lr=FULL_PHASE_LR)
image_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    image_optimizer, FULL_EPOCHS, eta_min=FULL_FINAL_LR)
phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    phase_optimizer, FULL_EPOCHS, eta_min=FULL_FINAL_LR)

output = OUTPUT_DIR / "full_600"
output.mkdir(parents=True, exist_ok=True)
history = []
started = time.time()

print(f"开始完整联合重建：size={size}, frames={FRAMES}, device={DEVICE}")
for epoch in trange(FULL_EPOCHS):
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
    image, field, phase = network.estimates()
    image = image[0, 0].cpu().numpy()
    field = field[0, 0].cpu().numpy()
    phase = phase[0, 0].cpu().numpy()

display_image = np.clip(image, 0, None)
display_image /= max(float(display_image.max()), 1e-8)
sio.savemat(output / "reconstruction.mat", {
    "image": image,
    "field": field,
    "phase": phase,
    "loss": np.asarray(history, dtype=np.float32),
})
torch.save(network.state_dict(), output / "model.pt")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
axes[0].imshow(display_image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("完整600×600重建物体")
axes[1].imshow(np.angle(field), cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[1].set_title("完整600×600系统相位")
axes[2].semilogy(history)
axes[2].set(title="训练损失", xlabel="epoch", ylabel="MSE")
for axis in axes[:2]:
    axis.set(xlabel="x (px)", ylabel="y (px)")
fig.savefig(output / "summary.png", dpi=180)
plt.close(fig)

(output / "summary.json").write_text(json.dumps({
    "size": size,
    "frames": FRAMES,
    "epochs": FULL_EPOCHS,
    "batch": FULL_BATCH,
    "device": DEVICE,
    "elapsed_seconds": time.time() - started,
    "final_mse": history[-1],
}, indent=2), encoding="utf-8")
print(f"完成：{output}")
