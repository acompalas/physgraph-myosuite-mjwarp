"""MyoHand DexHand definitions, GENUINELY INHERITING PhysGraph's real
DexHand base class (physgraph_envs/lib/envs/dexhands/base.py), loaded
standalone via importlib to bypass that package's IsaacGym/bps_torch-
coupled __init__.py chain (same technique used elsewhere in this port
-- see project notes). This gives us real inheritance (to_hand/to_dex/
n_dofs/n_bodies/reverse_mapping all come from the actual PhysGraph
class), not just an informally-mirrored interface."""
import importlib.util
import os

import numpy as np

# Locally: ~/physgraph-local/PhysGraph. On the pod: /data/PhysGraph.
# Override with the PHYSGRAPH_ROOT env var if neither matches.
_PHYSGRAPH_ROOT = os.environ.get(
    "PHYSGRAPH_ROOT",
    "/data/PhysGraph" if os.path.isdir("/data/PhysGraph") else os.path.expanduser("~/physgraph-local/PhysGraph"),
)
_DEXHAND_BASE_PATH = os.path.join(
    _PHYSGRAPH_ROOT, "physgraph_envs/lib/envs/dexhands/base.py"
)


def _load_dexhand_base():
    spec = importlib.util.spec_from_file_location("dexhand_base", _DEXHAND_BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DexHand


DexHand = _load_dexhand_base()


class MyoHandR(DexHand):
    def __init__(self):
        super().__init__()
        self.name = "myohand"
        self.side = "rh"

        # innermost body of each composite-joint split pair (see
        # scripts/cut_hand_ulna_root.py), matching pytorch_kinematics'
        # actual frame names in the preprocessed chain
        self.body_names = [
            "lunate_r_j1",          # wrist (after deviation + flexion)
            "firstmc_r_j1",         # thumb metacarpal (after cmc flex+abd)
            "proximal_thumb_r",     # thumb proximal phalanx (mp_flexion)
            "distal_thumb_r",       # thumb distal phalanx (ip_flexion)
            "THtip_r",              # thumb tip (site)
            "2proxph_r_j1", "midph2_r", "distph2_r", "IFtip_r",  # index
            "3proxph_r_j1", "midph3_r", "distph3_r", "MFtip_r",  # middle
            "4proxph_r_j1", "midph4_r", "distph4_r", "RFtip_r",  # ring
            "5proxph_r_j1", "midph5_r", "distph5_r", "LFtip_r",  # pinky
        ]

        self.dof_names = [
            "pro_sup_r", "deviation_r", "flexion_r",
            "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
            "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
            "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
            "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
            "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
        ]

        self.hand2dex_mapping = {
            "wrist": ["lunate_r_j1"],
            "thumb_proximal": ["firstmc_r_j1"],
            "thumb_intermediate": ["proximal_thumb_r"],
            "thumb_distal": ["distal_thumb_r"],
            "thumb_tip": ["THtip_r"],
            "index_proximal": ["2proxph_r_j1"],
            "index_intermediate": ["midph2_r"],
            "index_distal": ["distph2_r"],
            "index_tip": ["IFtip_r"],
            "middle_proximal": ["3proxph_r_j1"],
            "middle_intermediate": ["midph3_r"],
            "middle_distal": ["distph3_r"],
            "middle_tip": ["MFtip_r"],
            "ring_proximal": ["4proxph_r_j1"],
            "ring_intermediate": ["midph4_r"],
            "ring_distal": ["distph4_r"],
            "ring_tip": ["RFtip_r"],
            "pinky_proximal": ["5proxph_r_j1"],
            "pinky_intermediate": ["midph5_r"],
            "pinky_distal": ["distph5_r"],
            "pinky_tip": ["LFtip_r"],
        }
        # reverse_mapping is inherited from DexHand (a @staticmethod there)
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)

        self.contact_body_names = [
            "distal_thumb_r", "distph2_r", "distph3_r", "distph4_r", "distph5_r",
        ]
        self.bone_links = [
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20],
        ]
        self.weight_idx = {
            "thumb_tip": [4],
            "index_tip": [8],
            "middle_tip": [12],
            "ring_tip": [16],
            "pinky_tip": [20],
            "level_1_joints": [1, 5, 9, 13, 17],
            "level_2_joints": [2, 3, 6, 7, 10, 11, 14, 15, 18, 19],
        }
        self.self_collision = False
        self.relative_rotation = np.eye(3)
        self.relative_translation = np.zeros(3)
        # PID gains -- reference values from ArtiMANO (closest in style),
        # NOT yet tuned for MyoHand's own mass/scale
        self.Kp_rot = 0.3
        self.Ki_rot = 0.01
        self.Kd_rot = 0.005
        self.Kp_pos = 10
        self.Ki_pos = 0.003
        self.Kd_pos = 0.5

        # to_hand/to_dex/n_dofs/n_bodies all inherited from DexHand now

    def __str__(self):
        return "myohand_rh"


class MyoHandL(DexHand):
    def __init__(self):
        super().__init__()
        self.name = "myohand"
        self.side = "lh"

        self.body_names = [
            "lunate_l_j1",          # wrist (after deviation + flexion)
            "firstmc_l_j1",         # thumb metacarpal (after cmc flex+abd)
            "proximal_thumb_l",     # thumb proximal phalanx (mp_flexion)
            "distal_thumb_l",       # thumb distal phalanx (ip_flexion)
            "THtip_l",              # thumb tip (site)
            "2proxph_l_j1", "midph2_l", "distph2_l", "IFtip_l",  # index
            "3proxph_l_j1", "midph3_l", "distph3_l", "MFtip_l",  # middle
            "4proxph_l_j1", "midph4_l", "distph4_l", "RFtip_l",  # ring
            "5proxph_l_j1", "midph5_l", "distph5_l", "LFtip_l",  # pinky
        ]

        self.dof_names = [
            "pro_sup_l", "deviation_l", "flexion_l",
            "cmc_flexion_l", "cmc_abduction_l", "mp_flexion_l", "ip_flexion_l",
            "mcp2_flexion_l", "mcp2_abduction_l", "pm2_flexion_l", "md2_flexion_l",
            "mcp3_flexion_l", "mcp3_abduction_l", "pm3_flexion_l", "md3_flexion_l",
            "mcp4_flexion_l", "mcp4_abduction_l", "pm4_flexion_l", "md4_flexion_l",
            "mcp5_flexion_l", "mcp5_abduction_l", "pm5_flexion_l", "md5_flexion_l",
        ]

        self.hand2dex_mapping = {
            "wrist": ["lunate_l_j1"],
            "thumb_proximal": ["firstmc_l_j1"],
            "thumb_intermediate": ["proximal_thumb_l"],
            "thumb_distal": ["distal_thumb_l"],
            "thumb_tip": ["THtip_l"],
            "index_proximal": ["2proxph_l_j1"],
            "index_intermediate": ["midph2_l"],
            "index_distal": ["distph2_l"],
            "index_tip": ["IFtip_l"],
            "middle_proximal": ["3proxph_l_j1"],
            "middle_intermediate": ["midph3_l"],
            "middle_distal": ["distph3_l"],
            "middle_tip": ["MFtip_l"],
            "ring_proximal": ["4proxph_l_j1"],
            "ring_intermediate": ["midph4_l"],
            "ring_distal": ["distph4_l"],
            "ring_tip": ["RFtip_l"],
            "pinky_proximal": ["5proxph_l_j1"],
            "pinky_intermediate": ["midph5_l"],
            "pinky_distal": ["distph5_l"],
            "pinky_tip": ["LFtip_l"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)

        self.contact_body_names = [
            "distal_thumb_l", "distph2_l", "distph3_l", "distph4_l", "distph5_l",
        ]
        self.bone_links = [
            [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20],
        ]
        self.weight_idx = {
            "thumb_tip": [4],
            "index_tip": [8],
            "middle_tip": [12],
            "ring_tip": [16],
            "pinky_tip": [20],
            "level_1_joints": [1, 5, 9, 13, 17],
            "level_2_joints": [2, 3, 6, 7, 10, 11, 14, 15, 18, 19],
        }
        self.self_collision = False
        self.relative_rotation = np.eye(3)
        self.relative_translation = np.zeros(3)
        self.Kp_rot = 0.3
        self.Ki_rot = 0.01
        self.Kd_rot = 0.005
        self.Kp_pos = 10
        self.Ki_pos = 0.003
        self.Kd_pos = 0.5

    def __str__(self):
        return "myohand_lh"
