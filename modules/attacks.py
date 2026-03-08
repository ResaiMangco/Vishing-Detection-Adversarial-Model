from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttackResult:
    adversarial_inputs: torch.Tensor
    perturbations: torch.Tensor


class Attack(ABC):
    @abstractmethod
    def generate(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> AttackResult:
        raise NotImplementedError


# ===[[ Fast Gradient Sign Method (FGSM) Attack ]]===

@dataclass(frozen=True)
class FGSMAttackConfig:
    epsilon: float = 0.002
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    targeted: bool = False


class FGSMAttack(Attack):
    def __init__(
        self,
        config: Optional[FGSMAttackConfig] = None,
        loss_fn: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config or FGSMAttackConfig()
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

    def generate(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> AttackResult:
        original_mode = model.training
        model.eval()

        adversarial_inputs = inputs.detach().clone().requires_grad_(True)
        detached_labels = labels.detach().clone()

        logits = model(adversarial_inputs)
        loss = self.loss_fn(logits, detached_labels)

        gradient = torch.autograd.grad(loss, adversarial_inputs, retain_graph=False, create_graph=False)[0]
        direction = -1.0 if self.config.targeted else 1.0
        perturbations = direction * self.config.epsilon * gradient.sign()
        perturbed = adversarial_inputs + perturbations
        perturbed = perturbed.clamp(self.config.clamp_min, self.config.clamp_max).detach()

        if original_mode:
            model.train()

        return AttackResult(
            adversarial_inputs=perturbed,
            perturbations=(perturbed - inputs.detach()).detach(),
        )
