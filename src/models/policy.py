import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data

from src.models.GCN import GCNEncoder
from src.utils import StepSample, Observation


class Policy(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def update(self, step: StepSample):
        raise NotImplementedError

    def masked_action_distribution(
        self, logits: torch.Tensor, legal_actions: list[int]
    ) -> torch.Tensor:
        """Softmax over legal logits only; full-sized prob vector (zeros on illegal actions)."""
        device = logits.device
        dtype = logits.dtype
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=device)
        probs = torch.zeros(self.num_actions, dtype=dtype, device=device)
        if legal_idx.numel() > 0:
            legal_logits = logits.index_select(0, legal_idx)
            legal_probs = torch.softmax(legal_logits, dim=0)
            # scatter_(dim, index, src): index and src must be same shape; probs is 1-D
            probs.scatter_(0, legal_idx, legal_probs)

        return probs

    def log_prob_legal(self, logits: torch.Tensor, legal_actions: list[int], action: int) -> torch.Tensor:
        """log π(a|s) where π is softmax over logits restricted to legal_actions."""
        if not legal_actions:
            raise ValueError("log_prob_legal requires non-empty legal_actions")
        device = logits.device
        legal_idx = torch.tensor(legal_actions, dtype=torch.long, device=device)
        legal_logits = logits.index_select(0, legal_idx)
        log_p = F.log_softmax(legal_logits, dim=0)
        try:
            pos = legal_actions.index(action)
        except ValueError as exc:
            raise ValueError(f"action {action} not in legal_actions {legal_actions}") from exc
        return log_p[pos]


class ReinforcePolicy(Policy):
    """Encoder + head; REINFORCE update with masked softmax over legal actions."""

    def __init__(self, encoder: nn.Module, head: nn.Module, num_actions: int):
        super().__init__(num_actions)
        self.encoder = encoder
        self.head = head

    def forward(self, obs: Observation) -> torch.Tensor:
        emb = self.encoder(obs=obs)
        logits = self.head(emb)
        return logits

    def update(self, step: StepSample, advantage: torch.Tensor, learning_rate: float = 1e-3) -> float:
        """REINFORCE: gradient of -advantage * log π(a|s) (masked over legal actions)."""
        self.zero_grad(set_to_none=True)
        logits = self.forward(step.observation)
        log_pi = self.log_prob_legal(logits, step.observation.legal_actions, step.action)
        loss = -(advantage * log_pi)
        loss.backward()
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
        return float(loss.item())
