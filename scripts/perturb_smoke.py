"""CPU smoke test for the neurons/perturb engine (Problem 1 / feasibility solver).

Patches the single forward choke point (neurons.perturb.utils.logits_for_images) with a small
differentiable linear stub placed NEAR the decision boundary, so a handful of single-byte flips
suffice — the regime the miner targets. Exercises perturb() end-to-end (which runs find_feasible) and
the solver built against a Context directly.

Checks: normal m0>0 image -> in-band, grid-aligned k=1 flip; already-misclassified image keeps >=1
changed channel; byte invariant (max_step==1, on-grid) holds; find_feasible flips against a Context;
the unsafe-flip gate returns clean when nothing is envelope-safe and ALLOW_UNSAFE_FLIP=0, but flips
when =1.
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


def run(name, clean, t, m0_target, **env):
    """End-to-end perturb() (runs the hybrid) on the stub; report the returned candidate's metrics."""
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


def build_ctx(clean, t, m0_target=0.012, **env):
    """A Context wired to the stub, for exercising a single approach in isolation."""
    reset_env(**env)
    setup_stub(clean, t, m0_target)
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


def nz_grid_step(r, clean):
    diff = (r["cand"] - clean).reshape(-1)
    bytes_ = (diff * 255.0).round()
    on_grid = torch.allclose(diff, bytes_ / 255.0, atol=1e-6)
    return r["nz"], on_grid, int(bytes_.abs().max().item())


def main():
    clean = torch.randint(0, 256, (3, 64, 64)).float() / 255.0  # grid-aligned, like a PNG decode

    print("[1] perturb() end-to-end (find_feasible) — normal flip near boundary")
    nz, linf, grid, step, flip = run("feasible", clean, 7, 0.012)
    assert flip, "should flip"
    assert grid and step == 1, "must be grid-aligned, one byte per channel"
    assert 0.003 - 1e-9 <= linf <= 0.03 + 1e-9, "L_inf in band"

    print("[2] already-misclassified — keep >=1 channel, in band")
    nz2, linf2, grid2, step2, flip2 = run("alreadywrong", clean, 7, -0.012)
    assert flip2 and nz2 >= 1 and grid2 and step2 == 1
    assert linf2 >= 0.003 - 1e-9

    print("[3] find_feasible flips grid-aligned (built against a Context)")
    torch.manual_seed(100)
    ctx = build_ctx(clean, 11, 0.012)
    P.find_feasible(ctx)
    r = ctx.bank.result(ctx.allow_unsafe)
    assert r is not None, "find_feasible found no safe flip"
    ref, on_grid, mstep = nz_grid_step(r, clean)
    print(f"  [find_feasible] |S|={ref} on_grid={on_grid} max_step={mstep}")
    assert on_grid and mstep == 1, "find_feasible must keep a grid-aligned one-byte flip"

    print("[4] find_feasible_upgraded flips grid-aligned (built against a Context)")
    torch.manual_seed(100)
    ctx = build_ctx(clean, 11, 0.012)
    P.find_feasible_upgraded(ctx)
    r = ctx.bank.result(ctx.allow_unsafe)
    assert r is not None, "find_feasible_upgraded found no safe flip"
    ref, on_grid, mstep = nz_grid_step(r, clean)
    print(f"  [find_feasible_upgraded] |S|={ref} on_grid={on_grid} max_step={mstep}")
    assert on_grid and mstep == 1, "find_feasible_upgraded must keep a grid-aligned one-byte flip"

    print("[4b] perturb() end-to-end via PERTURB_SOLVER=upgraded")
    nzu, linfu, gridu, stepu, flipu = run("upgraded", clean, 7, 0.012, PERTURB_SOLVER="upgraded")
    assert flipu and gridu and stepu == 1 and 0.003 - 1e-9 <= linfu <= 0.03 + 1e-9

    print("[4c] Approach 2 (compress) shrinks |S| of a dense flip")
    A2 = importlib.import_module("neurons.perturb.approach2")
    torch.manual_seed(100)
    ctx = build_ctx(clean, 13, 0.012)
    _, move_dir, _ = U.loss_grad(ctx.model, ctx.clean, ctx.target_index, "ce")   # dense gradient-sign flip
    A2._eval(ctx, [move_dir * float(ctx.k_min)])
    r0 = ctx.bank.result(ctx.allow_unsafe)
    assert r0 is not None, "dense seed did not produce a safe flip"
    base = r0["nz"]
    A2.compress(ctx, repair_fn=None)
    r1 = ctx.bank.result(ctx.allow_unsafe)
    ref, on_grid, mstep = nz_grid_step(r1, clean)
    print(f"  [approach2] |S| {base} -> {ref}  (drop {100.0 * (base - ref) / max(1, base):.1f}%) "
          f"on_grid={on_grid} max_step={mstep}")
    assert on_grid and mstep == 1, "approach2 must keep a grid-aligned one-byte flip"
    assert ref < base, f"approach2 must strictly shrink |S| on the exact-gradient stub ({ref} !< {base})"

    print("[4d] perturb() end-to-end with PERTURB_APPROACH2=1 (feasible + compress)")
    nzc, linfc, gridc, stepc, flipc = run("a2", clean, 7, 0.012, PERTURB_APPROACH2="1")
    assert flipc and gridc and stepc == 1 and 0.003 - 1e-9 <= linfc <= 0.03 + 1e-9

    print("[5] unsafe-flip gate — kappa beyond the achievable margin swing makes nothing safe")
    nzg, _, _, _, flipg = run("gate-off", clean, 7, 0.005,
                              PERTURB_MINER_MARGIN_BUFFER="50.0", PERTURB_ALLOW_UNSAFE_FLIP="0")
    assert nzg == 0 and not flipg, "ALLOW_UNSAFE_FLIP=0 must return clean when nothing is safe"
    _, _, gg, sg, flipa = run("gate-on", clean, 7, 0.005,
                              PERTURB_MINER_MARGIN_BUFFER="50.0", PERTURB_ALLOW_UNSAFE_FLIP="1")
    assert flipa and gg and sg == 1, "ALLOW_UNSAFE_FLIP=1 must return any flip"

    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
