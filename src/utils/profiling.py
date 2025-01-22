import time

from torch import cuda
from torch.utils.tensorboard import SummaryWriter

from src.utils.train import convert_time_units


class LoopProfiler:
    _base_profiling_unit = "s"

    def __init__(
        self,
        writer: SummaryWriter,
        profiles: list[str],
        profiling_unit: str = "ms",
        cuda_sync: bool = True,
        profiling_name: str = "Profiling",
    ):
        self._writer: SummaryWriter = writer
        self._profiling_unit = profiling_unit
        self._profile_names: dict[str:str] = {
            profile: f"{profiling_name}/{profile}-{profiling_unit}" for profile in profiles
        }

        self._profile_start_times = {profile: None for profile in profiles}

        self._profile_times = {profile: None for profile in profiles}

        self.cuda_sync = cuda_sync

    def start(self, profile: str):
        if self.cuda_sync:
            cuda.synchronize()

        self._profile_start_times[profile] = time.perf_counter()

    def stop(self, profile: str):
        if self.cuda_sync:
            cuda.synchronize()

        elapsed_time = time.perf_counter() - self._profile_start_times[profile]
        self._profile_times[profile] = elapsed_time

    def log_all(self, step_num: int):
        if self.cuda_sync:
            cuda.synchronize()

        for profile, profiled_time in self._profile_times.items():
            if profiled_time is not None:
                profiled_time = convert_time_units(profiled_time, self._profiling_unit, self._base_profiling_unit)
                self._writer.add_scalar(self._profile_names[profile], profiled_time, step_num, new_style=True)
                self._profile_times[profile] = None
