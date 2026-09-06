# ---
# Copied verbatim from PhysGraph (github.com/acompalas/PhysGraph).
# Confirmed IsaacGym-free (traced 2026-09-06) -- reused as-is per
# the project's inherit-where-possible strategy.
# ---
import torch.nn as nn


class Identity(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        super().__init__()
        self._output_dim = input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        return x

    def get_optimizer_groups(self, *args, **kwargs):
        return [], []
