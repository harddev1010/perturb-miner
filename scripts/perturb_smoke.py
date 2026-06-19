"""CPU smoke test for the neurons/perturb engine.

Patches the single forward choke point (neurons.perturb.utils.logits_for_images) with a small
differentiable linear stub placed NEAR the decision boundary, so a handful of single-byte flips
suffice — the regime the miner targets. Exercises perturb() end-to-end plus the switches:
all three orchestrators, a single-algorithm PIPELINE, and the PERTURB_ALLOW_UNSAFE_FLIP gate.

Checks: normal m0>0 image -> sparse, in-band, grid-aligned k=1 flip; already-misclassified image
keeps >=1 changed channel; byte invariant (max_step==1, on-grid) holds; the unsafe-flip gate
returns clean when nothing is envelope-safe and ALLOW_UNSAFE_FLIP=0, but flips when =1.
"""
import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import neurons.perturb.constants as C
import neurons.perturb.utils as U
# `neurons.perturb.perturb` (submodule) is shadowed by the perturb() function re-exported in the
# package __init__, so grab the module object explicitly.
P = importlib.import_module("neurons.perturb.perturb")

D = 3 * 64 * 64
torch.manual_seed(0)
_M = torch.randn(D, 1000) * 0.2  # scale so a handful of single-byte flips can clear a small margin
_OFFSET = torch.zeros(1000)


def stub_logits(model, image_bchw):
    x = image_bchw.reshape(image_bchw.shape[0], -1)  # bypass the 480 resize/normalize
    return x @ _M - _OFFSET


def setup_stub(clean, t, m0_target):
    """Place class t at margin m0_target vs its nearest competitor at the clean image."""
    global _OFFSET
    raw = clean.reshape(-1) @ _M
    comp = (t + 1) % 1000
    desired = torch.full((1000,), -50.0)
    desired[t] = 0.0
    desired[comp] = -m0_target
    _OFFSET = raw - desired


device = torch.device("cpu")


def reset_env(**env):
    for k in list(os.environ):
        if k.startswith("PERTURB_"):
            del os.environ[k]
    for k, v in env.items():
        os.environ[k] = v
    importlib.reload(C)               # constants re-read env in place; utils/perturb hold the same object
    U.logits_for_images = stub_logits  # the one patch point all forwards funnel through


def run(name, clean, t, m0_target, expect_clean=False, **env):
    reset_env(**env)
    setup_stub(clean, t, m0_target)
    adv = P.perturb(model=None, clean=clean, target_index=t, epsilon=0.03, min_delta=0.003,
                    device=device, timeout_seconds=8.0)
    diff = adv - clean
    nz = int((diff.abs() > 1e-6).sum().item())
    linf = float(diff.abs().max().item())
    bytes_ = (diff.reshape(-1) * 255.0).round()
    on_grid = torch.allclose(diff.reshape(-1), bytes_ / 255.0, atol=1e-6)
    max_step = int(bytes_.abs().max().item()) if nz else 0
    flipped = int(stub_logits(None, adv.unsqueeze(0))[0].argmax().item()) != t
    print(f"  [{name}] nz={nz} linf={linf:.6f} (~{linf * 255:.2f}/255) on_grid={on_grid} "
          f"max_step={max_step} flipped={flipped}")
    return nz, linf, on_grid, max_step, flipped


def main():
    clean = torch.randint(0, 256, (3, 64, 64)).float() / 255.0  # grid-aligned, like a PNG decode
    FAST = {"PERTURB_FIND_FLIP_BUDGET": "1.5"}  # keep run_all/Square short on CPU

    print("[1] default (first_safe) — normal flip near boundary")
    nz, linf, grid, step, flip = run("flip", clean, 7, 0.012, **FAST)
    assert flip, "should flip"
    assert grid and step == 1, "must be grid-aligned, one byte per channel"
    assert 0.003 - 1e-9 <= linf <= 0.03 + 1e-9, "L_inf in band"

    print("[2] already-misclassified — keep >=1 channel, in band")
    nz2, linf2, grid2, step2, flip2 = run("alreadywrong", clean, 7, -0.012, **FAST)
    assert flip2 and nz2 >= 1 and grid2 and step2 == 1
    assert linf2 >= 0.003 - 1e-9

    print("[3] all three orchestrators flip (switchable by the one call-site name)")
    saved = P.find_flip_first_safe
    for orch in (P.find_flip_first_safe, P.find_flip_first_hit, P.find_flip_run_all):
        P.find_flip_first_safe = orch  # perturb() resolves this name at call time
        _, _, g, s, f = run(orch.__name__, clean, 11, 0.012, **FAST)
        assert f and g and s == 1, f"{orch.__name__} should flip grid-aligned"
    P.find_flip_first_safe = saved

    print("[4] single-algorithm PIPELINE still flips (comment-out simulation)")
    saved_pipe = P.PIPELINE
    for algo in (P.batched_multi_loss_qfgsm, P.quantized_pgd_one_byte,
                 P.quantized_saliency_greedy):
        P.PIPELINE = [algo]
        _, _, g, s, f = run(algo.__name__, clean, 11, 0.012,
                            PERTURB_ALLOW_UNSAFE_FLIP="1", **FAST)
        assert f and g and s == 1, f"{algo.__name__} alone should flip"
    P.PIPELINE = saved_pipe

    print("[5] unsafe-flip gate — kappa beyond the achievable margin swing makes nothing safe")
    # kappa=50 exceeds the largest swing any single-byte flip can produce here, so no candidate is 'safe'.
    nzg, _, _, _, flipg = run("gate-off", clean, 7, 0.005,
                              PERTURB_MINER_MARGIN_BUFFER="50.0", PERTURB_ALLOW_UNSAFE_FLIP="0", **FAST)
    assert nzg == 0 and not flipg, "ALLOW_UNSAFE_FLIP=0 must return clean when nothing is safe"
    _, _, gg, sg, flipa = run("gate-on", clean, 7, 0.005,
                              PERTURB_MINER_MARGIN_BUFFER="50.0", PERTURB_ALLOW_UNSAFE_FLIP="1", **FAST)
    assert flipa and gg and sg == 1, "ALLOW_UNSAFE_FLIP=1 must return any flip"

    print("[6] RMSE refiners — seed a DENSE flip, then each optim_rmse_* shrinks |S| (Bank-decoupled)")
    # Built directly against a Context (not the live orchestrator switch), so this stays valid no matter
    # which finder/refiners perturb()'s switch happens to call. Seed = the full CE-sign dense byte flip.
    def build_ctx(t):
        reset_env(PERTURB_FIND_FLIP_BUDGET="5.0")   # also re-patches U.logits_for_images with the stub
        setup_stub(clean, t, 0.012)
        cu8 = torch.round(clean.view(-1) * 255.0)
        m0, g0 = U.margin_and_grad(None, clean, t)
        dl = time.time() + 8.0
        return P.Context(
            model=None, device=device, clean=clean, clean_u8=cu8, shape=clean.shape,
            target_index=t, k_min=1, q=1.0 / 255.0, floor=0.003, cap=0.03, kappa=C.MARGIN_BUFFER,
            skip_roundtrip=C.SKIP_ROUNDTRIP, tf32_on=C.TF32_ON, envelope=False,
            allow_unsafe=C.ALLOW_UNSAFE_FLIP, deadline=dl, t_step=0.005,
            time_left=lambda: dl - time.time(), bank=P.Bank(), m0=m0, g0=g0.view(-1),
        )

    def seed_dense(ctx):                            # dense gradient-sign flip as the warm-start
        _, move_dir, _ = U.loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")
        P._eval_deltas(ctx, [move_dir * float(ctx.k_min)])

    def seed_prefix(ctx, k):                         # over-provisioned top-k flip (free channels remain to swap in)
        _, move_dir, score = U.loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")
        _, order, valid = U.build_sparse_order(score, move_dir, ctx.clean.view(-1))
        d = torch.zeros_like(ctx.clean_u8)
        idx = order[:min(int(k), valid)]
        d[idx] = move_dir[idx] * float(ctx.k_min)
        P._eval_deltas(ctx, [d])

    def nz_grid_step(r):                            # |S|, on-grid, max byte step of a Bank result's candidate
        diff = (r["cand"] - clean).reshape(-1)
        bytes_ = (diff * 255.0).round()
        on_grid = torch.allclose(diff, bytes_ / 255.0, atol=1e-6)
        return r["nz"], on_grid, int(bytes_.abs().max().item())

    # exchange needs free (unused, movable) channels to swap IN, so seed it over-provisioned, not fully dense.
    refiners = [
        ("prune", [P.optim_rmse_prune], seed_dense),
        ("exchange", [P.optim_rmse_exchange], lambda c: seed_prefix(c, 3000)),
        ("fmn_l2", [P.optim_rmse_fmn_l2], seed_dense),
        ("fab_l2", [P.optim_rmse_fab_l2], seed_dense),
        ("sigma_zero", [P.optim_rmse_sigma_zero], seed_dense),
        ("fmn_l0", [P.optim_rmse_fmn_l0], seed_dense),
        ("prune+exchange", [P.optim_rmse_prune, P.optim_rmse_exchange], seed_dense),
    ]
    for name, chain, seed in refiners:
        torch.manual_seed(100)
        ctx = build_ctx(13)
        seed(ctx)
        r0 = ctx.bank.result(ctx.allow_unsafe)
        assert r0 is not None, f"{name}: seed did not produce a safe flip"
        base = r0["nz"]
        for refiner in chain:
            refiner(ctx)
        r1 = ctx.bank.result(ctx.allow_unsafe)
        assert r1 is not None, f"{name} lost the flip"
        ref, on_grid, step = nz_grid_step(r1)
        print(f"  [{name}] |S| {base} -> {ref}  (drop {100.0 * (base - ref) / max(1, base):.1f}%) "
              f"on_grid={on_grid} max_step={step}")
        assert on_grid and step == 1, f"{name} must keep a grid-aligned one-byte flip"
        assert ref < base, f"{name} must strictly shrink |S| on the exact-gradient stub ({ref} !< {base})"

    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
