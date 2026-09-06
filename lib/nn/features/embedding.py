# ---
# Copied verbatim from PhysGraph (github.com/acompalas/PhysGraph).
# Confirmed IsaacGym-free (traced 2026-09-06) -- reused as-is per
# the project's inherit-where-possible strategy.
# ---
from torch.nn import Embedding as _Embedding


class Embedding(_Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = self.embedding_dim
