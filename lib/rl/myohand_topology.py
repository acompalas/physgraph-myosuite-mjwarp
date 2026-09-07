"""MyoHand-specific graph topology for the transformer network, replacing
PhysGraph's ManoTopology (network_builder_transformer_bih_graph_improve_
correct.py). Genuinely NEW code -- MANO's per-hand structure (28 nodes:
1 palm + 27 joints, 5 per finger except thumb's 7) doesn't match
MyoHand's real structure, verified directly against our own body_names
(envs/mjwarp_myohand_env.py) 2026-09-08: 21 nodes per hand (1 wrist +
4 joints x 5 fingers, uniform including thumb -- level_1/MCP, level_2a/
PIP, level_2b/DIP, tip).

Follows PhysGraph's exact method structure (dist_mat via Floyd-Warshall,
serial_mask, synergy_mask, centrality) and exact design conventions
(explicit high centrality for wrists, explicit zero centrality for
manipulated-object nodes), just with our own verified numbers:
  - 21 nodes/hand (not MANO's 28), offset 21 for LH (not 28)
  - 3 synergy groups (MCP/PIP/tip) -- a real, complete anatomical match,
    not a compromise: MyoHand's thumb has the same 3-joint (MCP/PIP/DIP)
    + tip structure as the other 4 fingers, just under different names
  - 2 object nodes (our two mugs, source+destination) instead of
    PhysGraph's tool+single-object -- our task is mug-to-mug pouring,
    not tool-mediated manipulation of one object
  - num_phys_nodes = 44 total (21+21+2), not PhysGraph's 58
"""
import torch
import torch.nn as nn


class MyoHandTopology(nn.Module):
    """
    44-node graph for bimanual MyoHand + 2 manipulated objects.
    RH: 0-20, LH: 21-41 (offset 21). Object nodes: 42 (source mug), 43 (dest mug).
    Per-hand layout (verified against envs/mjwarp_myohand_env.py body_names,
    2026-09-08): 0=wrist, then 5 fingers x 4 joints each
    (thumb=1-4, index=5-8, middle=9-12, ring=13-16, pinky=17-20),
    each finger = [MCP/level_1, PIP/level_2a, DIP/level_2b, tip].
    """
    def __init__(self):
        super().__init__()
        self.num_phys_nodes = 44
        self.register_buffer('dist_mat', self._build_dist_mat())
        self.register_buffer('serial_mask', self._build_serial_mask())
        self.register_buffer('synergy_mask', self._build_synergy_mask())
        self.register_buffer('centrality', self._build_centrality())

    def _build_dist_mat(self):
        dists = torch.full((self.num_phys_nodes, self.num_phys_nodes), 10, dtype=torch.long)
        dists.fill_diagonal_(0)

        def connect(i, j):
            dists[i, j] = 1
            dists[j, i] = 1

        def wire_hand(offset):
            palm = offset
            # each finger: palm -> MCP -> PIP -> DIP -> tip (4 joints, serial chain)
            finger_starts = [offset + 1, offset + 5, offset + 9, offset + 13, offset + 17]
            for start in finger_starts:
                connect(palm, start)
                connect(start, start + 1)
                connect(start + 1, start + 2)
                connect(start + 2, start + 3)

        wire_hand(0)
        wire_hand(21)
        mug_src, mug_dst = 42, 43
        connect(0, mug_src)
        connect(21, mug_src)
        connect(0, mug_dst)
        connect(21, mug_dst)
        connect(mug_src, mug_dst)

        for k in range(self.num_phys_nodes):
            for i in range(self.num_phys_nodes):
                for j in range(self.num_phys_nodes):
                    dists[i, j] = min(dists[i, j], dists[i, k] + dists[k, j])
        return dists.clamp(max=5)

    def _build_serial_mask(self):
        mask = torch.full((self.num_phys_nodes, self.num_phys_nodes), float('-inf'))
        mask[self._build_dist_mat() <= 1] = 0.0
        return mask

    def _build_synergy_mask(self):
        mask = torch.full((self.num_phys_nodes, self.num_phys_nodes), float('-inf'))
        mask.fill_diagonal_(0.0)
        # per-finger indices, RH offset (0): MCP=level_1, PIP=level_2a, tip
        mcps = [1, 5, 9, 13, 17]
        pips = [2, 6, 10, 14, 18]
        tips = [4, 8, 12, 16, 20]
        for grp in [mcps, pips, tips]:
            for x in [0, 21]:  # RH then LH
                indices = [i + x for i in grp]
                for i in indices:
                    for j in indices:
                        mask[i, j] = 0.0
        return mask

    def _build_centrality(self):
        degs = torch.ones(self.num_phys_nodes, dtype=torch.long)
        degs[0] = 5   # RH wrist
        degs[21] = 5  # LH wrist
        degs[42] = 0  # source mug
        degs[43] = 0  # dest mug
        return degs
