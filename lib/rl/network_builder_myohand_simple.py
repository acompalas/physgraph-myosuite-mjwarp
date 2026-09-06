"""Simple dict-observation network builder for the MyoHand pour task.

This is genuinely NEW code (not reused verbatim), but follows PhysGraph's
own real architectural pattern exactly -- see
lib/rl/network_builder_transformer_bih_graph_improve_correct.py's
BimanualHandToolPolicy/ResBiHDictObsBuilder for the reference: subclass
A2CBuilder.Network, override __init__/forward, return the same
(action_mu, sigma, value, None, None) 5-tuple rl_games expects, feed
obs['privileged'] directly into the value head (asymmetric actor-critic,
same as the real class does).

The one real difference: PhysGraph's BimanualHandToolPolicy uses their
full graph-transformer LATENT_ENCODER (LATENT_ENCODER class), which is
hardcoded to a DIFFERENT embodiment's exact dof/body counts (22 dofs,
27 bodies -- likely ArtiMANO) plus real BPS object-shape encoding we
don't compute. We use lib.nn.features.SimpleFeatureFusion/Identity
instead -- REAL, unmodified PhysGraph utility classes, just without the
transformer/attention/BPS machinery on top. Adopting the full transformer
network (with correctly-derived MyoHand dimensions and real BPS encoding)
is a deliberate, separate FUTURE milestone once a second hand does real
work -- see project notes.

Our own observation dimensions (bimanual-SHAPED: RH real + LH inert-zero,
same width each -- see envs/mjwarp_myohand_env.py's _compute_obs):
  proprioception: 164 (82 RH + 82 LH)
  privileged: 60 (30 RH + 30 LH)
  target: 60 (30 RH + 30 LH)
These are hardcoded directly, matching PhysGraph's own precedent of
hardcoding LATENT_ENCODER's dimensions rather than deriving them
dynamically -- consistent, not a shortcut unique to our code.
"""
import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder

from lib.nn.features import SimpleFeatureFusion, Identity


class SimpleHandPolicy(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        NetworkBuilder.BaseNetwork.__init__(self)
        actions_num = kwargs.pop("actions_num")

        self.feature_fusion = SimpleFeatureFusion(
            extractors={
                "proprioception": Identity(input_dim=164),
                "privileged": Identity(input_dim=60),
                "target": Identity(input_dim=60),
            },
            hidden_depth=3,
            hidden_dim=512,
            output_dim=256,
            activation="swish",
            add_input_activation=False,
            add_output_activation=False,
        )
        self.action_mu_head = nn.Linear(256, actions_num)
        self.value_head = nn.Linear(256 + 60, 1)  # fused features + raw privileged (60)
        self.sigma = nn.Parameter(-torch.ones(actions_num, dtype=torch.float), requires_grad=True)

    def forward(self, obs, pre_action=None):
        z = self.feature_fusion(obs)
        action_mu = self.action_mu_head(z)
        sigma = self.sigma
        value = self.value_head(torch.cat([z, obs["privileged"]], dim=-1))
        return (action_mu, sigma, value, None, None)


class SimpleDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = SimpleHandPolicy(self.params, **kwargs)
        return net
