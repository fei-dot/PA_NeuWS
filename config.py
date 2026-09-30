"""只放需要经常修改的实验参数。入口脚本按从上到下的顺序使用这些变量。"""

from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data919" / "2026-09-19_slm50_noll4_28_decay4x_reference_guided"
OUTPUT_DIR = PROJECT / "outputs" / "0928_clean_reconstruction"

FRAMES = 50
SLM_PHASE_SIGN = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260805

# 完整600×600 NeuWS联合重建。
FULL_EPOCHS = 400
FULL_BATCH = 8
FULL_IMAGE_LR = 1e-3
FULL_PHASE_LR = 2e-3
FULL_FINAL_LR = 1e-3
FULL_ZERNIKE_MODES = 28
FULL_PHASE_LAYERS = 4

# 中心相位估计 + 50帧全场Wiener重建。
CENTER_WINDOWS = (64,)
CENTER_EPOCHS = 800
CENTER_BATCH = 16
CENTER_IMAGE_LR = 1e-3
CENTER_PHASE_LR = 2e-3
CENTER_FINAL_LR = 1e-3
CENTER_ZERNIKE_MODES = 28
CENTER_PHASE_LAYERS = 4
WIENER_REGULARIZATION = 5e-3
