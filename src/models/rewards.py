class Reward:
    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        raise NotImplementedError


class SizeReward:
    def __init__(self, initial_size: int, initial_depth: int):
        self.weight = 1/initial_size

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        return self.weight * (prev_size - size)


class ZhuSizeReward:
    """Paper-like size reward: normalized per-step gain minus constant baseline."""

    def __init__(
        self,
        initial_size: int,
        initial_depth: int,
        baseline_per_step: float = 0.0,
        baseline_scale: float = 1.0,
    ):
        self.weight = 1 / initial_size
        self.baseline_per_step = float(baseline_per_step)
        self.baseline_scale = float(baseline_scale)
        self.last_gain = 0.0

    def set_baseline(self, baseline_per_step: float) -> None:
        self.baseline_per_step = float(baseline_per_step)

    def set_baseline_scale(self, baseline_scale: float) -> None:
        self.baseline_scale = float(baseline_scale)

    def __call__(self, size: int, depth: int, prev_size: int, prev_depth: int) -> float:
        gain = self.weight * (prev_size - size)
        self.last_gain = float(gain)
        return gain - self.baseline_scale * self.baseline_per_step


class DepthReward:
    def __init__(self, initial_depth: int, initial_size: int):
        self.weight = 1/initial_depth

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        return self.weight * (prev_depth - depth)


class ProductOfDiffReward:
    def __init__(self, initial_size: int, initial_depth: int):
        self.size_reward = SizeReward(initial_size, initial_depth)
        self.depth_reward = DepthReward(initial_size, initial_depth)

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        return (
            self.size_reward(size, depth, prev_size, prev_depth) *
            self.depth_reward(size, depth, prev_size, prev_depth)
        )


class LinearReward:
    def __init__(self, initial_size: int, initial_depth: int):
        self.size_weight = 1/initial_size
        self.depth_weight = 1/initial_depth

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        return self.size_weight * (prev_size - size) + self.depth_weight * (prev_depth - depth)

