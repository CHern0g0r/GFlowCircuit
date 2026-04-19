from src.models.policy import Policy
from src.utils import StepSample, Observation
import pyspiel
from torch.distributions import Categorical

OBS_STEP_IDX = 0
OBS_NUM_STEPS_IDX = 1
OBS_SIZE_IDX = 2
OBS_DEPTH_IDX = 3
OBS_REWARD_IDX = 4


def get_obs_dim_and_num_actions(num_steps: int, sample_circuit: str) -> tuple[int, int]:
    probe_game = pyspiel.load_game("circuit", {"num_steps": int(num_steps), "file_path": sample_circuit})
    probe_state = probe_game.new_initial_state()
    obs_dim = len(probe_state.observation_tensor(0))
    num_actions = int(probe_game.num_distinct_actions())
    return obs_dim, num_actions


def run_episode(
    file_path: str,
    num_steps: int,
    policy: Policy,
    sample_actions: bool,
) -> list[StepSample]:
    game = pyspiel.load_game("circuit", {"num_steps": num_steps, "file_path": file_path})
    state = game.new_initial_state()
    trajectory: list[StepSample] = []

    obs0 = Observation.from_state(state)
    initial_size = int(obs0.obs_tensor[OBS_SIZE_IDX])
    initial_depth = int(obs0.obs_tensor[OBS_DEPTH_IDX])

    while not state.is_terminal():
        obs = Observation.from_state(state)
        logits = policy(obs)
        probs = policy.masked_action_distribution(logits, obs.legal_actions)

        if sample_actions:
            action = int(Categorical(probs).sample())
        else:
            action = int(probs.argmax())

        state.apply_action(action)
        next_obs = Observation.from_state(state)
        reward = float(next_obs.obs_tensor[OBS_REWARD_IDX])
        trajectory.append(StepSample(
            observation=obs,
            action=action,
            probs=probs,
            reward=reward,
        ))

    final_obs = Observation.from_state(state)
    return {
        "trajectory": trajectory,
        "initial_size": initial_size,
        "initial_depth": initial_depth,
        "final_size": int(final_obs.obs_tensor[OBS_SIZE_IDX]),
        "final_depth": int(final_obs.obs_tensor[OBS_DEPTH_IDX]),
        "final_return": float(state.returns()[0]),
        "total_reward": float(sum(step.reward for step in trajectory)),
        "num_steps_taken": len(trajectory),
        "terminal": bool(state.is_terminal()),
    }