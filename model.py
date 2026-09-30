"""静态NeuWS网络。只保留当前真实实验实际使用的网络分支。"""

import math

import torch
import torch.nn as nn
from torch.fft import fft2, fftshift, irfftn, rfftn


NOLL = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2), 6: (2, 2),
    7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
    13: (4, -2), 14: (4, 4), 15: (4, -4), 16: (5, 1), 17: (5, -1), 18: (5, 3),
    19: (5, -3), 20: (5, 5), 21: (5, -5), 22: (6, 0), 23: (6, -2), 24: (6, 2),
    25: (6, -4), 26: (6, 4), 27: (6, -6), 28: (6, 6),
}


def radial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    m = abs(m)
    value = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** k * math.factorial(n - k)
            / (math.factorial(k)
               * math.factorial((n + m) // 2 - k)
               * math.factorial((n - m) // 2 - k))
        )
        value = value + coefficient * rho ** (n - 2 * k)
    return value


def square_zernike_basis(modes: int, size: int) -> torch.Tensor:
    axis = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    rho = torch.sqrt(xx.square() + yy.square())
    theta = torch.atan2(yy, xx)
    basis = []
    for index in range(1, modes + 1):
        n, m = NOLL[index]
        angular = (torch.cos(m * theta) if m > 0 else
                   torch.sin(abs(m) * theta) if m < 0 else torch.ones_like(rho))
        scale = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
        value = scale * radial(n, m, rho) * angular
        if index == 1:
            value = torch.ones_like(value)
        else:
            value = value - value.mean()
            value = value / value.square().mean().sqrt().clamp_min(1e-8)
        basis.append(value)
    return torch.stack(basis, dim=-1)


def linear_convolution(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    size = image.shape[-1]
    shape = (2 * size, 2 * size)
    output = irfftn(
        rfftn(image, dim=(-2, -1), s=shape)
        * rfftn(kernel, dim=(-2, -1), s=shape),
        dim=(-2, -1), s=shape,
    )
    half = size // 2
    return output[..., half:-half, half:-half]


class PixelRenderer(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )


class ImagePatch(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.features = nn.Parameter(0.1 * torch.randn(size, size, 32))
        self.renderer = PixelRenderer()

    def forward(self) -> torch.Tensor:
        axis = torch.linspace(0.5 / self.size, 1 - 0.5 / self.size,
                              self.size, device=self.features.device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        positions = torch.stack((y.flatten(), x.flatten()), dim=1) * self.size
        indices = positions.long()
        weights = positions - indices
        x0 = indices[:, 0].clamp(0, self.size - 1)
        y0 = indices[:, 1].clamp(0, self.size - 1)
        x1 = (x0 + 1).clamp(max=self.size - 1)
        y1 = (y0 + 1).clamp(max=self.size - 1)
        wx, wy = weights[:, :1], weights[:, 1:]
        sampled = (
            self.features[y0, x0] * (1 - wx) * (1 - wy)
            + self.features[y0, x1] * wx * (1 - wy)
            + self.features[y1, x0] * (1 - wx) * wy
            + self.features[y1, x1] * wx * wy
        )
        return self.renderer(sampled).reshape(1, 1, self.size, self.size)


class ObjectImage(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        half = size // 2
        self.patches = nn.ModuleList([ImagePatch(half) for _ in range(4)])

    def forward(self) -> torch.Tensor:
        top_left, top_right, bottom_left, bottom_right = [patch() for patch in self.patches]
        top = torch.cat((top_left, top_right), dim=-1)
        bottom = torch.cat((bottom_left, bottom_right), dim=-1)
        return torch.cat((top, bottom), dim=-2)


class StaticNeuWS(nn.Module):
    def __init__(self, size: int, zernike_modes: int, phase_layers: int):
        super().__init__()
        self.object = ObjectImage(size)
        self.register_buffer("phase_coordinates", square_zernike_basis(zernike_modes, size)[None])

        layers = [nn.Linear(zernike_modes, 32)]
        for _ in range(phase_layers):
            layers += [nn.Linear(32, 32), nn.LeakyReLU(inplace=True)]
        layers.append(nn.Linear(32, 2))
        self.aberration = nn.Sequential(*layers)

    def estimates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.object()
        amplitude_phase = self.aberration(self.phase_coordinates).permute(0, 3, 1, 2)
        amplitude = amplitude_phase[:, :1]
        phase = amplitude_phase[:, 1:]
        field = amplitude * torch.exp(1j * phase)
        return image, field, phase

    def forward(self, slm_field: torch.Tensor):
        image, field, phase = self.estimates()
        psf = fftshift(fft2(field * slm_field, norm="forward"), dim=(-2, -1)).abs().square()
        psf = psf / psf.sum(dim=(-2, -1), keepdim=True)
        psf = psf.flip(-2, -1)
        prediction = linear_convolution(image, psf)
        return prediction.squeeze(1), psf, image, field, phase

