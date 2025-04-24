from pathlib import Path

import numpy as np


def gaussian_delays(n_pmts: int, save_path: str | Path, mean: float, std: float, seed: int) -> str:
    save_path = Path(save_path)
    if save_path.is_dir():
        save_path = save_path / f"cable_delays_gaussian-mean={mean}_std={std}_seed={seed}.txt"

    generator = np.random.default_rng(seed)
    delays = generator.normal(loc=mean, scale=std, size=n_pmts)
    delays[0] = 0.0
    np.savetxt(save_path, delays)

    return str(save_path)
