"""CPU smoke test for the v2 two-phase miner. Patches logits_for_images with a small differentiable
linear stub placed NEAR the decision boundary (a few single-byte flips suffice — the regime the miner
targets) so we exercise perturb() end-to-end: growth -> first-flip -> prune/repair/swap -> finalize.
Checks: flip case lands a sparse in-band k=1 candidate; already-wrong case keeps >=1 channel;
grid-aligned byte invariant; legacy + no-opt paths still produce valid flips."""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import neurons.miner as miner

D = 3 * 64 * 64
torch.manual_seed(0)
_M = torch.randn(D, 1000) * 0.2  # scale so a handful of single-byte flips can clear a ~0.01 margin
_OFFSET = torch.zeros(1000)


def stub_logits(model, image_bchw):
    x = image_bchw.reshape(image_bchw.shape[0], -1)  # bypass the 480 resize/normalize
    return x @ _M - _OFFSET


def setup_stub(clean, t, m0_target):
    """Place class t at margin m0_target vs its nearest competitor at the clean image."""
    global _OFFSET
    raw = (clean.reshape(-1) @ _M)
    comp = (t + 1) % 1000
    desired = torch.full((1000,), -50.0)
    desired[t] = 0.0
    desired[comp] = -m0_target
    _OFFSET = raw - desired


device = torch.device("cpu")


def run(name, clean, t, m0_target, **env):
    # Config is read at module import, so reload under the variant's env to actually exercise it.
    for k in list(os.environ):
        if k.startswith("PERTURB_"):
            del os.environ[k]
    for k, v in env.items():
        os.environ[k] = v
    importlib.reload(miner)
    miner.logits_for_images = stub_logits  # re-apply the patch (reload reset it)
    setup_stub(clean, t, m0_target)
    adv = miner.perturb(model=None, clean=clean, target_index=t, epsilon=0.03, min_delta=0.003,
                        device=device, timeout_seconds=8.0)
    diff = adv - clean
    nz = int((diff.abs() > 1e-6).sum().item())
    linf = float(diff.abs().max().item())
    bytes_ = (diff.reshape(-1) * 255.0).round()
    on_grid = torch.allclose(diff.reshape(-1), bytes_ / 255.0, atol=1e-6)
    max_step = int(bytes_.abs().max().item())
    flipped = int(stub_logits(None, adv.unsqueeze(0))[0].argmax().item()) != t
    print(f"  [{name}] nz={nz} linf={linf:.6f} (~{linf*255:.2f}/255) on_grid={on_grid} "
          f"max_step={max_step} flipped={flipped}")
    return nz, linf, on_grid, max_step, flipped


def main():
    clean = torch.randint(0, 256, (3, 64, 64)).float() / 255.0  # grid-aligned, like a PNG decode

    print("[1] tf32_on (default), full v2 flow — normal flip near boundary")
    nz, linf, grid, step, flip = run("flip", clean, 7, 0.012)
    assert flip, "should flip"
    assert grid and step == 1, "must be grid-aligned k=1"
    assert 0.003 - 1e-9 <= linf <= 0.03 + 1e-9, "L_inf in band"

    print("[2] already-misclassified — keep >=1 channel, in band")
    nz2, linf2, grid2, step2, flip2 = run("alreadywrong", clean, 7, -0.012)
    assert flip2 and nz2 >= 1 and grid2 and step2 == 1
    assert linf2 >= 0.003 - 1e-9

    print("[3] tf32_unknown (envelope) — collapses to off on CPU, still flips")
    _, _, grid3, step3, flip3 = run("unknown", clean, 11, 0.012,
                                    PERTURB_VALIDATOR_NUMERIC_MODE="tf32_unknown")
    assert flip3 and grid3 and step3 == 1

    print("[4] legacy grow-to-safe (first-flip-stop off) — still flips")
    _, _, grid4, step4, flip4 = run("legacy", clean, 11, 0.012, PERTURB_FIRST_FLIP_STOP="0")
    assert flip4 and grid4 and step4 == 1

    print("[5] all prune/refine off — valid flip (regression floor)")
    _, _, grid5, step5, flip5 = run("noopt", clean, 11, 0.012, PERTURB_GROUP_DELETE_ENABLE="0",
                                    PERTURB_GRAD_REFRESH_ENABLE="0", PERTURB_PRUNE_ENABLE="0")
    assert flip5 and grid5 and step5 == 1

    print("[6] legacy PERTURB_TF32_ENVELOPE maps to a numeric mode when the new var is unset")
    run("legacy-env", clean, 11, 0.012, PERTURB_TF32_ENVELOPE="1")
    assert miner._NUMERIC_MODE == "tf32_unknown", miner._NUMERIC_MODE
    run("legacy-env-off", clean, 11, 0.012, PERTURB_TF32_ENVELOPE="0")
    assert miner._NUMERIC_MODE == "tf32_off", miner._NUMERIC_MODE

    print("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
