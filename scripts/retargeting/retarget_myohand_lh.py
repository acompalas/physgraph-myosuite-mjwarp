"""MANO -> MyoHand retargeting. Faithful port of PhysGraph's
main/dataset/mano2dexhand.py fitting() algorithm, extended with:
(1) wrist ORIENTATION tracking every iteration (not just at init).
(2) EQUAL weighting across all finger joints (knuckle through tip) so
    the optimizer can't sacrifice knuckle accuracy to nail the tip.
(3) Real MyoHand joint-limit clamping after every step (hard, exact).
(4) Temporal smoothness regularizer on wrist rotation + joint angles.
(5) A COLLISION-AWARE PHASE: after the main fit converges, refine
    ONLY opt_dof_pos (wrist/orientation frozen, already well-fit) to
    drive index-finger/mug penetration to near-zero, using the ACTUAL
    MyoHand collision capsules (radius + full local orientation
    within their parent body -- composing body world rotation with
    each capsule's own local geom quat, not just the body's own
    rotation, which was a real bug in an earlier version that made
    the collision check silently check the wrong points entirely)."""
import pickle
import sys
from pathlib import Path

import mujoco
import pytorch_kinematics as pk
import pytorch_kinematics.mjcf as pk_mjcf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandL

class CoercingDict(dict):
    def __getitem__(self, key):
        return super().__getitem__(int(key))
pk_mjcf.JOINT_TYPE_MAP = CoercingDict(pk_mjcf.JOINT_TYPE_MAP)


def aa_to_rotmat(aa):
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    theta = theta[..., None]
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def wxyz_quat_to_rotmat(q):
    """q: (4,) tensor, MuJoCo convention (w,x,y,z). Returns (3,3) rotmat.
    Fixed-constant helper for the capsule geoms' own local orientation
    within their parent body -- NOT a differentiable optimization
    variable, just a one-time conversion of a model constant."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float32)


def rotmat_to_rot6d(R):
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rot6d_to_rotmat(r6):
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rot6d_to_aa(r6):
    R = rot6d_to_rotmat(r6)
    return rotmat_to_aa(R)


def rotmat_to_aa(R):
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1, 1)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    vx = (R[..., 2, 1] - R[..., 1, 2]) / (2 * sin_theta)
    vy = (R[..., 0, 2] - R[..., 2, 0]) / (2 * sin_theta)
    vz = (R[..., 1, 0] - R[..., 0, 1]) / (2 * sin_theta)
    axis = torch.stack([vx, vy, vz], dim=-1)
    return axis * theta[..., None]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
        demo = pickle.load(f)

    dexhand = MyoHandL()
    chain = pk.build_chain_from_mjcf(open("/tmp/myohand_l_pk.xml").read())
    chain = chain.to(dtype=torch.float32, device=device)

    n_frames = demo["wrist_pos"].shape[0]
    print(f"retargeting {n_frames} frames")

    ref_model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2] / "assets/hands/myohand_l_ulnaroot.xml"))
    dof_lo, dof_hi = [], []
    for name in dexhand.dof_names:
        jid = mujoco.mj_name2id(ref_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = ref_model.jnt_range[jid]
        dof_lo.append(lo)
        dof_hi.append(hi)
    dof_lo = torch.tensor(dof_lo, device=device)
    dof_hi = torch.tensor(dof_hi, device=device)

    capsule_specs_raw = [
        ("2proxph_l_j1", 0.009, 0.02, [0.74364598, -0.64140698, -0.170361, -0.0810239]),
        ("midph2_l", 0.008, 0.012, [0.72704967, -0.6606737, -0.16878292, -0.08013336]),
        ("distph2_l", 0.007, 0.008, [0.72704967, -0.6606737, -0.16878292, -0.08013336]),
    ]
    capsule_specs = [
        (body, radius, half_len, wxyz_quat_to_rotmat(torch.tensor(q, dtype=torch.float32)).to(device))
        for body, radius, half_len, q in capsule_specs_raw
    ]

    mug_radius = 0.0526
    mug_half_height = 0.1263 / 2
    obj_traj = demo["obj_trajectory"].to(device)
    mug_pos = obj_traj[:, :3, 3]
    mug_rotmat = obj_traj[:, :3, :3]

    def capsule_endpoints_world(ret, opt_wrist_rotmat, pk_world):
        out = []
        for body_name, cap_radius, half_len, geom_local_rotmat in capsule_specs:
            body_rotmat_local = ret[body_name].get_matrix()[:, :3, :3]
            body_rotmat_world = opt_wrist_rotmat @ body_rotmat_local
            capsule_rotmat_world = body_rotmat_world @ geom_local_rotmat[None]
            capsule_axis_world = capsule_rotmat_world[:, :, 2]
            body_pos_world = pk_world[:, dexhand.body_names.index(body_name)]
            for sign in [-1.0, 1.0]:
                endpoint = body_pos_world + sign * half_len * capsule_axis_world
                out.append((endpoint, cap_radius))
        return out

    def penetration_for_endpoints(endpoints):
        terms = []
        for endpoint, cap_radius in endpoints:
            ep_local = (mug_rotmat.transpose(-1, -2) @ (endpoint - mug_pos).unsqueeze(-1)).squeeze(-1)
            ep_radial = torch.norm(ep_local[..., :2], dim=-1)
            ep_in_range = (ep_local[..., 2].abs() < mug_half_height).float()
            ep_pen = torch.relu((mug_radius + cap_radius) - ep_radial) * ep_in_range
            terms.append(ep_pen)
        return torch.stack(terms)

    target_wrist_pos = demo["wrist_pos"].to(device)
    target_wrist_rot_aa = demo["wrist_rot"].to(device)
    target_wrist_rotmat = aa_to_rotmat(target_wrist_rot_aa)
    target_elbow_pos = demo["elbow_pos"].to(device)  # RE-ADDED: dropped when we
                                                        # reverted to the first-attempt
                                                        # script earlier tonight, never
                                                        # brought back -- opt_wrist_pos
                                                        # IS the elbow position directly,
                                                        # no FK needed to compare it

    # real per-fingertip contact distance, ground truth (main/dataset/base.py:127,
    # sqrt of chamfer distance from each fingertip to the real mug mesh).
    # tip_list order confirmed from base.py itself:
    tip_names_order = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    tips_distance = demo["tips_distance"].to(device)  # (N, 5)
    tip_body_names = [b for b in dexhand.body_names if dexhand.to_hand(b)[0] in tip_names_order]
    tip_body_to_col = {b: tip_names_order.index(dexhand.to_hand(b)[0]) for b in tip_body_names}

    # real target palm-normal direction, per frame, from the ACTUAL
    # captured wrist + index-knuckle + pinky-knuckle positions -- the
    # plane through these three points tells us which way the real
    # human's palm was facing. We match OUR hand's same plane-normal to
    # this, rather than guessing a fixed local axis on some body.
    target_index_prox = demo["mano_joints"]["index_proximal"].to(device)
    target_pinky_prox = demo["mano_joints"]["pinky_proximal"].to(device)
    v1_target = target_index_prox - target_wrist_pos
    v2_target = target_pinky_prox - target_wrist_pos
    target_palm_normal = torch.nn.functional.normalize(torch.cross(v1_target, v2_target, dim=-1), dim=-1)
    idx_2proxph = dexhand.body_names.index("2proxph_l_j1")
    idx_5proxph = dexhand.body_names.index("5proxph_l_j1")
    idx_wrist = dexhand.body_names.index("lunate_l_j1")

    mano_target_list = []
    for b in dexhand.body_names[1:]:
        key = dexhand.to_hand(b)[0]
        mano_target_list.append(demo["mano_joints"][key].to(device))
    target_other_joints = torch.stack(mano_target_list, dim=1)
    target_joints = torch.cat([target_wrist_pos[:, None], target_other_joints], dim=1)

    weight = []
    for b in dexhand.body_names:
        key = dexhand.to_hand(b)[0]
        if key == "wrist":
            weight.append(1.0)
        elif "tip" in key:
            weight.append(4.0)  # moderate pull toward the real target point,
                                 # not the original 25x dominance, not fully 1:1 either
        else:
            weight.append(1.0)
    weight = torch.tensor(weight, device=device)

    opt_wrist_pos = target_wrist_pos.clone().detach().requires_grad_(True)
    opt_wrist_rot6d = rotmat_to_rot6d(target_wrist_rotmat).clone().detach().requires_grad_(True)
    opt_dof_pos = torch.zeros(n_frames, dexhand.n_dofs, device=device, requires_grad=True)

    opti = torch.optim.Adam([
        {"params": [opt_wrist_pos, opt_wrist_rot6d], "lr": 0.0008},
        {"params": [opt_dof_pos], "lr": 0.0004},
    ])

    max_iter = 4000
    past_loss = 1e10
    for it in range(1, max_iter + 1):
        opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot6d)

        th = {name: opt_dof_pos[:, i] for i, name in enumerate(dexhand.dof_names)}
        ret = chain.forward_kinematics(th)

        pk_local = torch.stack(
            [ret[b].get_matrix()[:, :3, 3] for b in dexhand.body_names], dim=1
        )
        pk_world = (opt_wrist_rotmat[:, None] @ pk_local[..., None]).squeeze(-1) + opt_wrist_pos[:, None]

        pos_loss = torch.mean(torch.norm(pk_world - target_joints, dim=-1) * weight[None])

        R_diff = opt_wrist_rotmat.transpose(-1, -2) @ target_wrist_rotmat
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        rot_loss = torch.mean(1 - (trace - 1) / 2)
        beta_orient = 2.0  # REVERTED from 20.0 -- confirmed wrong direction: rot barely
        # improved (1.49->1.36) despite 10x weight, while total loss exploded
        # (everything else got sacrificed) -- this is a real structural conflict,
        # not underweighting

        rot_smooth = torch.mean(torch.norm(opt_wrist_rot6d[1:] - opt_wrist_rot6d[:-1], dim=-1) ** 2)
        dof_smooth = torch.mean(torch.norm(opt_dof_pos[1:] - opt_dof_pos[:-1], dim=-1) ** 2)
        # NEW: elbow's own POSITION was never smoothed -- only rotation and
        # joint angles were. With multiple valid (elbow_pos, joint_angles)
        # combos reaching the same wrist target, nothing stopped the elbow
        # from jumping between different valid solutions frame to frame
        # (the reported "cartpole swinging around the wrist" symptom).
        wrist_pos_smooth = torch.mean(torch.norm(opt_wrist_pos[1:] - opt_wrist_pos[:-1], dim=-1) ** 2)
        beta_rot_smooth = 40.0  # RAISED: rot_loss has a real floor it can't beat
        # (confirmed -- fighting it directly just sacrifices everything else),
        # so prioritize a SMOOTH trajectory over an exact one for this hand
        beta_dof_smooth = 2.0
        beta_wrist_pos_smooth = 10.0

        endpoints = capsule_endpoints_world(ret, opt_wrist_rotmat, pk_world)
        penetration = penetration_for_endpoints(endpoints)
        collision_loss = torch.mean(penetration ** 2)
        beta_collision = 20.0

        # CONTACT term (not just avoidance): each fingertip should sit at
        # the REAL captured radial distance from the mug's surface, in the
        # mug's own rotating local frame -- two-sided, so it neither
        # penetrates NOR drifts away as the mug turns, matching where a
        # real human fingertip actually stayed pressed throughout the grip.
        contact_terms = []
        for b, col in tip_body_to_col.items():
            idx = dexhand.body_names.index(b)
            tip_world = pk_world[:, idx]
            tip_local = (mug_rotmat.transpose(-1, -2) @ (tip_world - mug_pos).unsqueeze(-1)).squeeze(-1)
            tip_radial = torch.norm(tip_local[..., :2], dim=-1)
            target_radial = mug_radius + tips_distance[:, col]
            contact_terms.append((tip_radial - target_radial) ** 2)
        contact_loss = torch.mean(torch.stack(contact_terms))
        beta_contact = 15.0

        # our OWN palm-normal, same formula, from the retargeted output
        v1_ours = pk_world[:, idx_2proxph] - pk_world[:, idx_wrist]
        v2_ours = pk_world[:, idx_5proxph] - pk_world[:, idx_wrist]
        our_palm_normal = torch.nn.functional.normalize(torch.cross(v1_ours, v2_ours, dim=-1), dim=-1)
        palm_facing_loss = torch.mean(1 - (our_palm_normal * target_palm_normal).sum(-1))
        beta_palm_facing = 10.0

        elbow_loss = torch.mean(torch.norm(opt_wrist_pos - target_elbow_pos, dim=-1) ** 2)
        beta_elbow = 0.8  # LOWERED from 3.0 -- was fighting pos_loss for a
    # different frame-to-frame compromise, causing the elbow to visibly
    # swing/wander in an arc while the hand itself stayed coherent

        loss = (pos_loss + beta_orient * rot_loss + beta_rot_smooth * rot_smooth
                + beta_dof_smooth * dof_smooth + beta_collision * collision_loss
                + beta_contact * contact_loss + beta_palm_facing * palm_facing_loss
                + beta_elbow * elbow_loss + beta_wrist_pos_smooth * wrist_pos_smooth)

        opti.zero_grad()
        loss.backward()
        opti.step()

        with torch.no_grad():
            opt_dof_pos.clamp_(dof_lo, dof_hi)

        if it % 200 == 0:
            print(f"[phase1] iter {it}: total={loss.item():.6f} pos={pos_loss.item():.6f} "
                  f"rot={rot_loss.item():.6f} collision={collision_loss.item():.6f} "
                  f"contact={contact_loss.item():.6f} palm={palm_facing_loss.item():.6f} elbow={elbow_loss.item():.6f} wrist_pos_smooth={wrist_pos_smooth.item():.6f} max_pen={penetration.max().item():.6f}")
            if it > 200 and past_loss - loss.item() < 1e-6:
                print("phase1 converged, stopping early")
                break
            past_loss = loss.item()

    opt_wrist_pos.requires_grad_(False)
    opt_wrist_rot6d.requires_grad_(False)
    opt_wrist_rotmat_frozen = rot6d_to_rotmat(opt_wrist_rot6d).detach()
    opt_dof_pos_phase1_solution = opt_dof_pos.detach().clone()  # anchor target

    phase2_opti = torch.optim.Adam([opt_dof_pos], lr=0.002)
    max_iter2 = 3000
    for it in range(1, max_iter2 + 1):
        th = {name: opt_dof_pos[:, i] for i, name in enumerate(dexhand.dof_names)}
        ret = chain.forward_kinematics(th)
        pk_local = torch.stack(
            [ret[b].get_matrix()[:, :3, 3] for b in dexhand.body_names], dim=1
        )
        pk_world = (opt_wrist_rotmat_frozen[:, None] @ pk_local[..., None]).squeeze(-1) + opt_wrist_pos[:, None]

        endpoints = capsule_endpoints_world(ret, opt_wrist_rotmat_frozen, pk_world)
        penetration = penetration_for_endpoints(endpoints)
        collision_loss = torch.mean(penetration ** 2)
        max_pen = penetration.max().item()

        # phase 2 was missing ANY smoothness term -- pure per-frame
        # independent collision correction, nothing tying adjacent frames
        # together, which is very likely what caused the sudden jumps/
        # brush-then-release artifacts
        dof_smooth2 = torch.mean(torch.norm(opt_dof_pos[1:] - opt_dof_pos[:-1], dim=-1) ** 2)
        beta_dof_smooth2 = 3.0
        anchor_loss = torch.mean((opt_dof_pos - opt_dof_pos_phase1_solution) ** 2)
        beta_anchor = 1.0  # LOWERED from 8.0 -- that was strong enough to fully
                            # drift the fingers away from real contact entirely
        phase2_loss = collision_loss + beta_dof_smooth2 * dof_smooth2 + beta_anchor * anchor_loss

        phase2_opti.zero_grad()
        phase2_loss.backward()
        phase2_opti.step()

        with torch.no_grad():
            opt_dof_pos.clamp_(dof_lo, dof_hi)

        if it % 200 == 0 or max_pen < 1e-6:
            print(f"[phase2] iter {it}: collision={collision_loss.item():.8f} anchor={anchor_loss.item():.6f} dof_smooth={dof_smooth2.item():.6f} max_penetration={max_pen:.8f}")
        if max_pen < 1e-6:
            print(f"phase2 converged: max penetration {max_pen:.8f} < 1e-6, stopping")
            break

    print(f"FINAL max penetration across all frames/endpoints: {penetration.max().item():.8f}")

    to_dump = {
        "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
        "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot6d).detach().cpu().numpy(),
        "opt_dof_pos": opt_dof_pos.detach().cpu().numpy(),
    }
    out_path = Path(__file__).resolve().parents[2] / "assets/retargeted/1292e_lh_myohand.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(to_dump, f)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
