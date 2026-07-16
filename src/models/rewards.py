class Reward:
    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        raise NotImplementedError

    def set_baseline(self, baseline_per_step: float) -> None:
        self.baseline_per_step = float(baseline_per_step)

    def set_baseline_scale(self, baseline_scale: float) -> None:
        self.baseline_scale = float(baseline_scale)


class SizeReward(Reward):
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
    def __init__(self, initial_size: int, initial_depth: int):
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


class DrillsSizeDepthReward(Reward):
    """DRiLLS reward table adapted to size/depth circuit metrics."""

    def __init__(
        self,
        initial_size: int,
        initial_depth: int,
        depth_constraint_ratio: float = 1.0,
    ):
        self.initial_size = int(initial_size)
        self.initial_depth = int(initial_depth)
        self.depth_constraint_ratio = float(depth_constraint_ratio)
        self.depth_constraint = float(self.initial_depth) * self.depth_constraint_ratio
        self.last_size_delta = 0
        self.last_depth_delta = 0
        self.last_constraint_met = False

    @staticmethod
    def _sign_improvement(previous: int, current: int) -> int:
        if current < previous:
            return 1
        if current == previous:
            return 0
        return -1

    def __call__(self, size: int, depth: int, prev_size: int, prev_depth: int) -> float:
        size_improvement = self._sign_improvement(int(prev_size), int(size))
        depth_improvement = self._sign_improvement(int(prev_depth), int(depth))
        constraint_met = float(depth) <= self.depth_constraint

        self.last_size_delta = size_improvement
        self.last_depth_delta = depth_improvement
        self.last_constraint_met = bool(constraint_met)

        if constraint_met:
            return float({1: 3, 0: 0, -1: -1}[size_improvement])
        if depth_improvement == 1:
            return float({1: 3, 0: 2, -1: 1}[size_improvement])
        if depth_improvement == 0:
            return float({1: 2, 0: 0, -1: -2}[size_improvement])
        return float({1: -1, 0: -2, -1: -3}[size_improvement])


class DiffOfProductReward(Reward):
    def __init__(self,
                 initial_size: int,
                 initial_depth: int,
                 c_size: float = 1,
                 c_depth: float = 1,
                 baseline_scale: float = 1.0):
        self.c_size = c_size
        self.c_depth = c_depth
        self.baseline_scale = baseline_scale
        self.baseline_per_step = 0.0
        self.weight = 1/(initial_size**self.c_size * initial_depth**self.c_depth)

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        gain = self.weight * (
            prev_size**self.c_size * prev_depth**self.c_depth -
            size**self.c_size * depth**self.c_depth
        )
        self.last_gain = float(gain)
        return gain - self.baseline_scale * self.baseline_per_step

class LinearReward(Reward):
    def __init__(self,
                 initial_size: int,
                 initial_depth: int,
                 c_size: float = 1,
                 c_depth: float = 1,
                 baseline_scale: float = 1.0):
        self.size_weight = 1/initial_size
        self.depth_weight = 1/initial_depth
        self.c_size = c_size
        self.c_depth = c_depth
        self.baseline_scale = baseline_scale
        self.baseline_per_step = 0.0

    def __call__(self, size: int, depth: int,
                 prev_size: int, prev_depth: int) -> float:
        gain = (
            self.c_size * self.size_weight * (prev_size - size) +
            self.c_depth * self.depth_weight * (prev_depth - depth)
        )
        self.last_gain = float(gain)
        return gain - self.baseline_scale * self.baseline_per_step
