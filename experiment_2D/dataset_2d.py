import dataclasses
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclasses.dataclass
class ToyDatasetConfig:
    dataset_size: int = 200
    """Number of samples."""

    mode: str = "step"
    """Dataset mode: 'step', 'piecewise', 'random', or 'circle_line'."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


class Toy1DDataset(Dataset):
    """Synthetic 1D dataset for implicit vs explicit policy toy examples."""

    def __init__(self, config: ToyDatasetConfig) -> None:
        self.dataset_size = config.dataset_size
        self.mode = config.mode
        self.seed = config.seed

        self.rng = np.random.RandomState(self.seed)
        self._generate_data()

    def _generate_data(self):
        # Uniformly sample x ∈ [0,1]
        # 将随机数范围从 [0,1) 转换为 [-1,1)
        self._coordinates = (self.rng.rand(self.dataset_size, 1) * 2 - 1).astype(np.float32)
        self._targets = self._generate_y(self._coordinates)

    def _generate_y(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "step":
            return (x >= 0).astype(np.float32)

        elif self.mode == "piecewise":
            y = np.zeros_like(x)
            mask1 = x < -0.33
            y[mask1] = 2 * x[mask1]
            mask2 = (x >= -0.33) & (x < 0.33)
            y[mask2] = -2 * (x[mask2] + 0.33) + (2*-0.33)
            mask3 = x >= 0.33
            y[mask3] = 0.5 + 2 * (x[mask3] - 0.33)
            return y.astype(np.float32)

        elif self.mode == "random":
            return self.rng.rand(len(x), 1).astype(np.float32)
        
        elif self.mode == "circle_line":
            r = 0.7
            x_flat = x.flatten()  # 保证是一维
            y = np.zeros((len(x_flat), 3), dtype=np.float32)
            mask_circle = (x_flat >= -r) & (x_flat <= 0)
            y_upper = np.sqrt(np.maximum(0.0, r**2 - x_flat[mask_circle]**2))
            y[mask_circle, 1] = y_upper
            mask_circle = (x_flat >= -r) & (x_flat <= r)
            y_lower = -np.sqrt(np.maximum(0.0, r**2 - x_flat[mask_circle]**2))
            y[mask_circle, 2] = y_lower
            return y

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ---------------------------
    # 与 CoordinateRegression 对齐的接口
    # ---------------------------

    def exclude(self, coordinates: np.ndarray) -> None:
        """Exclude the given coordinates, if present, and resample new ones."""
        mask = (self._coordinates == coordinates[:, None]).all(-1).any(0)
        num_matches = mask.sum()
        while mask.sum() > 0:
            # 重新采样这些点
            self._coordinates[mask] = self.rng.rand(mask.sum(), 1).astype(np.float32)
            self._targets[mask] = self._generate_y(self._coordinates[mask])
            mask = (self._coordinates == coordinates[:, None]).all(-1).any(0)
        print(f"Resampled {num_matches} data points.")

    def get_target_bounds(self) -> np.ndarray:
        """Return per-dimension target min/max."""
        # 这里返回 y 的范围 [0,1]
        if self.mode == 'circle_line':
            return np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        else:
            return np.array([[0.0], [1.0]], dtype=np.float32)

    # ---------------------------
    # Dataset 接口
    # ---------------------------
    @property
    def coordinates(self) -> np.ndarray:
        """Return x values."""
        return self._coordinates

    @property
    def targets(self) -> np.ndarray:
        """Return y values."""
        return self._targets

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._coordinates[idx]),
            torch.from_numpy(self._targets[idx]),
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # cfg = ToyDatasetConfig(dataset_size=200, seed=0, mode="piecewise")
    # dataset = Toy1DDataset(cfg)

    # xs = dataset.coordinates.flatten()
    # ys = dataset.targets.flatten()

    # plt.scatter(xs, ys, marker="x", c="blue", alpha=0.6)
    # plt.title(f"Toy1DDataset - {cfg.mode}")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # print("Target bounds:", dataset.get_target_bounds())

    # 测试集值函数
    cfg = ToyDatasetConfig(dataset_size=200, seed=42, mode="circle_line")
    dataset = Toy1DDataset(cfg)

    # 随机取一个样本看 shape
    x, y = dataset[np.random.randint(len(dataset))]
    print("x =", x.shape)
    print("y =", y.shape)

    xs = dataset.coordinates.flatten()
    ys = dataset.targets  # shape (N, 3) for circle_with_line

    plt.figure()

    if ys.ndim == 1 or ys.shape[1] == 1:
        # 普通 1D 情况
        plt.scatter(xs, ys.flatten(), marker="x", c="blue", alpha=0.6)

    else:
        # 分别绘制 line / upper_arc / lower_arc
        plt.scatter(xs, ys[:, 0], marker="x", c="black", alpha=0.6, label="line")
        plt.scatter(xs, ys[:, 1], marker="o", c="blue", alpha=0.6, label="upper arc")
        plt.scatter(xs, ys[:, 2], marker="o", c="green", alpha=0.6, label="lower arc")

    plt.title(f"Toy1DDataset - {dataset.mode}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
