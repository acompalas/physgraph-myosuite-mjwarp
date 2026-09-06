# ---
# Copied verbatim from PhysGraph (github.com/acompalas/PhysGraph).
# Confirmed IsaacGym-free (traced 2026-09-06) -- reused as-is per
# the project's inherit-where-possible strategy.
# ---
from abc import ABC, abstractmethod

from pytorch_lightning import LightningModule


class BasePolicy(ABC, LightningModule):
    is_sequence_policy: bool = False

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward the NN.
        """
        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Given obs, return action.
        """
        pass
