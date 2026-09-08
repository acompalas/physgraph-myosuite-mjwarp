"""Full transformer network for MyoHand, reproducing PhysGraph's real
graph-transformer architecture (network_builder_transformer_bih_graph_
improve_correct.py's LATENT_ENCODER/BimanualHandToolPolicy) with
correctly-derived MyoHand dimensions and mujoco-warp's real API,
replacing the earlier SimpleHandPolicy/SimpleDictObsBuilder placeholder
network. Genuinely NEW code where MyoHand/mujoco-warp-specific (all
tokenizer dims, field slicing/reshaping, action_dim), but reuses
PhysGraph's real, confirmed embodiment-agnostic pieces verbatim via
import: CustomMultiheadAttention, TransformerLayer, MLP_new,
PhysicallyGroundedBias (works with zero modification when passed our
own MyoHandTopology -- confirmed 2026-09-08).

Field-by-field dimension derivation, tokenizer scope, and the full
token-sequence/idx_map structure are documented in the project notes
(memory) -- summary here: our real per-hand joint count is 20 (not
MANO's 27; note real MANO itself is 21 keypoints/hand, matching us --
the 27/28 figure is PhysGraph's own ArtiMANO-specific embodiment, not
real human-hand anatomy), giving smaller tokenizer/target dims
throughout. rlink_tokenizer(243), rh/lh_target_tokenizer(76), and the
type/side/id embeddings are all confirmed genuinely UNUSED in
PhysGraph's own real forward() pass (declared in __init__, never
called) -- not reproduced here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.rl.network_builder_transformer_bih_graph_improve_correct import (
    CustomMultiheadAttention, TransformerLayer, MLP_new, PhysicallyGroundedBias,
)
from lib.rl.myohand_topology import MyoHandTopology
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


class MyoHandMLPAction(nn.Module):
    """Real MLP_action, but with our own action_dim=48 (9 PID wrist +
    39 real MyoHand muscles) instead of PhysGraph's ArtiMANO-specific 56."""
    def __init__(self, action_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 1024), nn.SiLU(),
            nn.Linear(1024, 512), nn.SiLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MyoHandLatentEncoder(nn.Module):
    """Real LATENT_ENCODER, reproduced for MyoHand's verified dimensions.
    See module docstring + project notes for the full derivation."""
    def __init__(self, action_dim: int = 48, d_model: int = 64, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # tokenizers -- real MLP_new, our own verified input dims
        self.rh_tokenizer = MLP_new(141)
        self.lh_tokenizer = MLP_new(141)
        self.tool_tokenizer = MLP_new(129)
        self.obj_tokenizer = MLP_new(129)
        self.rh_curr_tokenizer = MLP_new(67)
        self.lh_curr_tokenizer = MLP_new(67)
        self.rlink_curr_tokenizer = MLP_new(9)
        self.llink_curr_tokenizer = MLP_new(9)

        self.policy_token_param = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.policy_token_param, std=0.02)

        self.rh_tok_ln = nn.LayerNorm(64)
        self.lh_tok_ln = nn.LayerNorm(64)
        self.tool_tok_ln = nn.LayerNorm(64)
        self.obj_tok_ln = nn.LayerNorm(64)

        self.topo = MyoHandTopology()
        self.bias_gen = PhysicallyGroundedBias(self.topo, num_heads)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, nhead=num_heads, dim_feedforward=512, dropout=0.0, activation="relu")
            for _ in range(4)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # idx_map: our real 46 physical tokens -> 44 graph nodes (each
        # wrist's hand+curr token pair maps to the SAME single node,
        # matching PhysGraph's own real convention)
        idx_map = torch.cat([
            torch.tensor([0, 0]),                    # RH wrist: hand + curr -> node 0
            torch.arange(1, 21),                     # RH links (20 joints)
            torch.tensor([21, 21]),                  # LH wrist: hand + curr -> node 21
            torch.arange(22, 42),                     # LH links (20 joints)
            torch.tensor([42, 43]),                   # src mug, dst mug
        ])
        self.register_buffer("idx_map", idx_map)

    def build_rh_token(self, x):
        return self.rh_tok_ln(self.rh_tokenizer(x)).unsqueeze(1)

    def build_lh_token(self, x):
        return self.lh_tok_ln(self.lh_tokenizer(x)).unsqueeze(1)

    def build_tool_token(self, x):
        return self.tool_tok_ln(self.tool_tokenizer(x)).unsqueeze(1)

    def build_object_token(self, x):
        return self.obj_tok_ln(self.obj_tokenizer(x)).unsqueeze(1)

    def build_rh_curr_token(self, x):
        return self.rh_tok_ln(self.rh_curr_tokenizer(x))

    def build_lh_curr_token(self, x):
        return self.lh_tok_ln(self.lh_curr_tokenizer(x))

    def build_rlink_curr_token(self, x):
        return self.rh_tok_ln(self.rlink_curr_tokenizer(x))

    def build_llink_curr_token(self, x):
        return self.lh_tok_ln(self.llink_curr_tokenizer(x))

    def build_policy_token(self, batch_size, device):
        return self.policy_token_param.view(1, 1, -1).expand(batch_size, 1, -1)

    def forward(self, obs):
        # === slice the 3 real observation streams (matches PhysGraph's
        # own real convention: one concatenated tensor per modality,
        # covering both hands, sliced internally here) ===
        r_prop = obs["proprioception"][:, :82]
        l_prop = obs["proprioception"][:, 82:]
        r_priv = obs["privileged"][:, :60]
        l_priv = obs["privileged"][:, 60:]
        r_curr = obs["target"][:, None, :380]
        l_curr = obs["target"][:, None, 380:]

        # proprioception: q/cosq/sinq/base_state
        r_q, r_cosq, r_sinq, r_base = r_prop[:, :23], r_prop[:, 23:46], r_prop[:, 46:69], r_prop[:, 69:82]
        l_q, l_cosq, l_sinq, l_base = l_prop[:, :23], l_prop[:, 23:46], l_prop[:, 46:69], l_prop[:, 69:82]

        # privileged: dq/obj_pos/obj_quat/obj_vel/obj_ang_vel/obj_com/obj_weight/tip_force
        r_dq, r_opos, r_oquat = r_priv[:, :23], r_priv[:, 23:26], r_priv[:, 26:30]
        r_ovel, r_oangvel, r_ocom = r_priv[:, 30:33], r_priv[:, 33:36], r_priv[:, 36:39]
        r_oweight, r_tipforce = r_priv[:, 39:40], r_priv[:, 40:60]
        l_dq, l_opos, l_oquat = l_priv[:, :23], l_priv[:, 23:26], l_priv[:, 26:30]
        l_ovel, l_oangvel, l_ocom = l_priv[:, 30:33], l_priv[:, 33:36], l_priv[:, 36:39]
        l_oweight, l_tipforce = l_priv[:, 39:40], l_priv[:, 40:60]

        rh_obs = torch.cat([r_q, r_cosq, r_sinq, r_base, r_dq, r_opos, r_oquat, r_ovel, r_oangvel, r_ocom, r_tipforce], dim=-1)
        lh_obs = torch.cat([l_q, l_cosq, l_sinq, l_base, l_dq, l_opos, l_oquat, l_ovel, l_oangvel, l_ocom, l_tipforce], dim=-1)

        tool_obs = torch.cat([r_oweight, r_curr[:, 0, 252:]], dim=-1)   # obj_weight + src BPS
        obj_obs = torch.cat([l_oweight, l_curr[:, 0, 252:]], dim=-1)    # obj_weight + dst BPS

        # target/curr: wrist(23) + object(23) + obj_to_joints(21) for the
        # curr token; per-joint tracking (60*3) for the links tokens
        r_wrist_obj = r_curr[:, :, :46]           # wrist block (0:23) + obj block (203:226) -- see note below
        # NOTE: wrist and object blocks are NOT contiguous in the real
        # target layout (joints block sits between them) -- build
        # rh_curr_obs by explicit concat, not a single contiguous slice
        r_curr_obs = torch.cat([
            r_curr[:, :, 0:23], r_curr[:, :, 203:226], r_curr[:, :, 226:247],
        ], dim=-1)
        l_curr_obs = torch.cat([
            l_curr[:, :, 0:23], l_curr[:, :, 203:226], l_curr[:, :, 226:247],
        ], dim=-1)

        r_links_obs = r_curr[:, 0, 23:203]   # delta_joints_pos+joints_vel+delta_joints_vel, (B,180)
        l_links_obs = l_curr[:, 0, 23:203]
        B = r_links_obs.shape[0]
        # reshape (B,180) -> (B,3,20,3) [3 fields, 20 joints, xyz] -> per-joint token (B,20,9)
        r_links_obs = r_links_obs.view(B, 3, 20, 3)
        r_links_obs = torch.stack([r_links_obs[:, 0], r_links_obs[:, 1], r_links_obs[:, 2]], dim=2).reshape(B, 20, 9)
        l_links_obs = l_links_obs.view(B, 3, 20, 3)
        l_links_obs = torch.stack([l_links_obs[:, 0], l_links_obs[:, 1], l_links_obs[:, 2]], dim=2).reshape(B, 20, 9)

        device = rh_obs.device

        rh_hand_tok = self.build_rh_token(rh_obs)
        lh_hand_tok = self.build_lh_token(lh_obs)
        tool_tok = self.build_tool_token(tool_obs)
        obj_tok = self.build_object_token(obj_obs)
        rh_curr_tok = self.build_rh_curr_token(r_curr_obs)
        lh_curr_tok = self.build_lh_curr_token(l_curr_obs)
        rh_links_tok = self.build_rlink_curr_token(r_links_obs)
        lh_links_tok = self.build_llink_curr_token(l_links_obs)
        policy_tok = self.build_policy_token(B, device)

        tokens = torch.cat([
            rh_hand_tok, rh_curr_tok, rh_links_tok,
            lh_hand_tok, lh_curr_tok, lh_links_tok,
            tool_tok, obj_tok,
        ], dim=1)
        tokens = torch.cat([policy_tok, tokens], dim=1)

        attn_bias_base = self.bias_gen(B, device)
        idx_map = self.idx_map
        attn_bias = attn_bias_base[:, :, idx_map, :][:, :, :, idx_map]
        attn_bias = F.pad(attn_bias, (1, 0, 1, 0), value=0.0)

        deg_base = self.topo.centrality.to(device)
        deg = deg_base[idx_map]
        deg = deg.clamp(min=0, max=self.bias_gen.cls_centrality_bias.num_embeddings - 1)
        b = self.bias_gen.cls_centrality_bias(deg)
        b = b.t().unsqueeze(0)
        attn_bias[:, :, 0, 1:] += self.bias_gen.cls_centrality_scale * b
        attn_bias[:, :, 1:, 0] += self.bias_gen.cls_centrality_scale * b

        for layer in self.layers:
            tokens = layer(tokens, src_mask=attn_bias)
        tokens = self.final_norm(tokens)

        return tokens[:, 0, :]


class MyoHandTransformerNetwork(A2CBuilder.Network):
    """Real BimanualHandToolPolicy pattern, for MyoHand. Wraps
    MyoHandLatentEncoder + action/value heads. Returns the standard
    4-tuple (mu, sigma, value, None), matching ModelA2CContinuousLogStd
    -- NOT PhysGraph's own 5-tuple (specific to their residual-checkpoint
    wrapper we don't use), same fix as the earlier SimpleHandPolicy."""
    def __init__(self, params, **kwargs):
        NetworkBuilder.BaseNetwork.__init__(self)
        action_dim = kwargs.get("actions_num", 48)
        self.encoder = MyoHandLatentEncoder(action_dim=action_dim)
        self.action_head = MyoHandMLPAction(action_dim=action_dim)
        self.value_head = nn.Linear(64, 1)
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def is_rnn(self):
        return False

    def forward(self, input_dict):
        obs = input_dict["obs"]
        policy_out = self.encoder(obs)
        mu = self.action_head(policy_out)
        value = self.value_head(policy_out)
        sigma = torch.nan_to_num(self.sigma, nan=-1.0)
        sigma = torch.clamp(sigma, -5.0, 2.0)
        return mu, sigma, value, None


class MyoHandTransformerBuilder(A2CBuilder):
    """Real network-builder wrapper, registered as
    'myohand_transformer_actor_critic' in train.py."""
    def build(self, name, **kwargs):
        return MyoHandTransformerNetwork(self.params, **kwargs)
