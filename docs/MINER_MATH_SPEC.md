# Perturb Miner — Mathematical Specification & Analysis Brief

**Purpose.** Self-contained writeup of the current miner's attack (`neurons/miner.py`, "dynamic
chunked engine") for review by a mathematician/optimization expert. It states the exact objective and
constraints, reduces the scoring to a one-variable problem, gives the algorithm as pseudocode with the
math behind each step, and ends with the assumptions we want stress-tested and the open questions.

All formulas below are transcribed from the live code: validator scoring from
`neurons/validator.py::verify_and_score`, scoring constants from `perturbnet/constants.py`, the attack
from `neurons/miner.py::perturb`.

---

## 1. Setup and notation

| Symbol | Meaning | Value / source |
|---|---|---|
| $x_0$ | clean image, flattened | $x_0\in[0,1]^n$ |
| $n$ | input dimension | $n=3HW$; default $H=W=64\Rightarrow n=12288$ (`PERTURB_IMAGE_SIZE=64`) |
| $f$ | classifier logits | $f:[0,1]^n\to\mathbb R^{1000}$, EfficientNetV2-L, **fixed** ImageNet-1k weights |
| $T$ | preprocessing | fixed differentiable transform (resize to 480², normalize); $f(x)=\mathrm{net}(T(x))$ |
| $t$ | true class index | attack must force $\arg\max_j f(x)_j\neq t$ |
| $q$ | unit step | $q=1/255\approx 3.9216\times10^{-3}$ |
| $\delta$ | perturbation | $\delta=x_{adv}-x_0$ |
| $S$ | changed-coordinate set | $S=\{i:\delta_i\neq0\}$, $|S|$ = sparsity |
| $\underline\epsilon,\ \bar\epsilon$ | L∞ band | $\underline\epsilon=0.003$, $\bar\epsilon=\min(\epsilon_{\text{syn}},0.03)$ |

**Key structural fact (the bridge).** Perturbation and *all* scoring happen in the $64\times64$ space,
but the network sees a $480\times480$ tensor produced by a fixed differentiable resize. Gradients
$\nabla_x m$ therefore already carry the resize Jacobian: one flipped $64\times64$ channel becomes a
smooth $\sim\!7.5\,\text{px}$ blob in model space. This is *not* a cross-model transfer problem — the
validator runs the **same weights**. The only train/serve gap is floating-point (TF32 / GPU / library),
handled by $\kappa$ and the TF32 envelope (§6.6).

**CW margin.** The single scalar that drives everything:
$$
m(x)\;=\;f(x)_t\;-\;\max_{j\neq t} f(x)_j,\qquad
m(x)<0 \iff \arg\max_j f(x)_j\neq t \ \ (\textbf{misclassified / "flipped"}).
$$
Its input gradient is the boundary normal $g=\nabla_x m(x)=\nabla_x f_t-\nabla_x f_{j^\star}$ where
$j^\star=\arg\max_{j\neq t}f(x)_j$.

---

## 2. Validator scoring (exact)

Let $\delta=x_{adv}-x_0$ (both sides decode from the **same PNG bytes**, so quantization is shared).
`SPEED_WEIGHT = 0`, so speed is irrelevant. The score is **0** unless *all* gates pass:

1. shape ok, $x_{adv}\in[0,1]^n$;
2. $\arg\max f(x_{adv})\neq t$ (else `label_match_with_original`);
3. $\|\delta\|_\infty \ge \underline\epsilon=0.003$ (else `below_min_delta`);
4. $\|\delta\|_\infty \le \bar\epsilon$ (else `above_max_delta`);
5. $\mathrm{SSIM}(x_0,x_{adv})\ge 0.98$;
6. $\mathrm{PSNR}(x_0,x_{adv})\ge 38\,$dB (miner enforces; validator's `min_psnr_db` may be 0).

If all pass, with $\mathrm{RMSE}(\delta)=\sqrt{\tfrac1n\sum_i\delta_i^2}$, $w_\infty=0.7$, $w_r=0.3$:
$$
r_\infty=\mathrm{clip}\!\Big(\tfrac{\|\delta\|_\infty-\underline\epsilon}{\bar\epsilon-\underline\epsilon},0,1\Big),\quad
s_\infty=(1-r_\infty)^2,
$$
$$
r_{\mathrm{rmse}}=\mathrm{clip}\!\Big(\tfrac{\mathrm{RMSE}(\delta)}{\bar\epsilon},0,1\Big),\quad
s_{\mathrm{rmse}}=(1-r_{\mathrm{rmse}})^2,
$$
$$
\boxed{\ \text{score}=\frac{w_\infty s_\infty+w_r s_{\mathrm{rmse}}}{w_\infty+w_r}\ }
$$

### 2.1 Reduction to a one-variable problem (the crux)

If the attack changes exactly $|S|$ channels, **each by a single byte** ($|\delta_i|=q$, i.e. $k=1$):

- $\|\delta\|_\infty=q$ — **constant in $|S|$** $\Rightarrow$ $s_\infty$ is a constant:
  $$
  r_\infty=\frac{q-\underline\epsilon}{\bar\epsilon-\underline\epsilon}=0.034132,\qquad s_\infty=0.93289 .
  $$
- $\mathrm{RMSE}=q\sqrt{|S|/n}$, so
  $$
  \boxed{\ \text{score}(|S|)\;=\;0.653023\;+\;0.3\Big(1-0.130719\sqrt{|S|/n}\Big)^{2}\ }\qquad(k=1)
  $$
  monotonically **decreasing** in $|S|$. Numerically ($n=12288$):

  | $\lvert S\rvert$ | 1 | 10 | 50 | 100 | 200 | 369 (≈3%) |
  |---|---|---|---|---|---|---|
  | score | **0.9523** | 0.9508 | 0.9480 | 0.9460 | 0.9431 | 0.9396 |

- **Why $k=1$ is mandatory:** moving to $k=2$ ($\|\delta\|_\infty=2/255$) gives $s_\infty=0.6734$, capping the
  score near $0.77$ regardless of sparsity. So the L∞ term forces single-byte steps; sparsity is the
  only remaining lever and it is a *gentle* one (1 → 369 channels costs only $\approx0.013$).
- **SSIM/PSNR are non-binding** in this regime: at $|S|=369$, $\mathrm{PSNR}\approx63\,$dB, $\mathrm{SSIM}\approx1$.

> **Therefore the entire problem is:** find the **minimum-cardinality** sign pattern of single-byte
> flips that (a) misclassifies and (b) survives numeric transfer ($m\le-\kappa$ under the TF32
> envelope), within a wall-clock budget. This is **minimum-$\ell_0$ adversarial perturbation with a
> fixed per-coordinate magnitude $q$** — combinatorial / NP-hard in general.

---

## 3. Optimization problem

$$
\min_{S\subseteq[n],\ \sigma\in\{\pm1\}^{S}} \ |S|
\quad\text{s.t.}\quad
m\!\Big(\Pi_{[0,1]}\big(x_0 - q\textstyle\sum_{i\in S}\sigma_i e_i\big)\Big)\le-\kappa,
$$
where $e_i$ is the $i$-th unit vector, $\Pi_{[0,1]}$ is the byte-clamp, and $\kappa\ge0$ is a transfer
cushion. The miner restricts the natural sign to the descent sign $\sigma_i=-\mathrm{sign}(g_i)$ and
solves this greedily with re-linearization. (Exact form: byte-space, §6.5.)

---

## 4. Algorithm — pseudocode

Three phases: **(A)** dynamic chunked greedy growth with re-linearization; **(B)** binary-search
backward prune; **(C)** authoritative finalize. Tunables (code defaults) in brackets.

```
INPUT: model f, clean x0∈[0,1]^n, true index t, band [ε_lo, ε_hi], budget B, κ
CONST: q = 1/255
       ALPHA   = 0.30      # fraction of remaining margin to close per chunk   [PERTURB_CHUNK_ALPHA]
       P_STEP  = 0.0025    # per-chunk cap, fraction of n                       [PERTURB_CHUNK_P_STEP]
       RHO     = 0.15      # per-chunk L2 cap                                   [PERTURB_CHUNK_RHO]
       b_max   = min( P_STEP·n , (RHO/q)^2 )           # ≈ min(30, 1463) = 30 channels/chunk
       κ       = 0.005 if TF32-envelope else 0.01      # [PERTURB_KAPPA_RESID / PERTURB_MINER_MARGIN_BUFFER]

# work in integer bytes; a coordinate, once changed, is never moved again ⇒ ‖δ‖∞ ≡ q exactly
u  ← round(255·x0)                  # byte image
C  ← ∅                              # set of changed coordinates  (= S)
(m, g) ← margin_and_grad(x0)        # CW margin + gradient at the CLEAN point   (§6.1–6.2)

# ---------- PHASE A: chunked growth ----------
while time_left > 2·t_step:
    if first iteration done:
        (m, g) ← margin_and_grad(u/255)            # RE-LINEARIZE at current perturbed point
    if C ≠ ∅ and current image is flipped (m<0, in band):
        bank as best_soft
        if m ≤ −κ AND envelope_margin(u/255) ≤ −κ:      # transfer-safe ⇒ stop (§6.6)
            best_safe ← current; break

    # rank feasible coordinates by salience |g_i|
    feasible ← { i ∉ C : coordinate can still move by −sign(g_i) without leaving [0,255] }
    sort feasible by |g_i| descending → order ;  C_b ← cumsum(|g_order|)     # prefix sums
    R ← m + κ                                          # remaining margin to overcome
    b_goal ← smallest b such that q·C_b ≥ ALPHA·R      # close ALPHA of remaining margin   (§6.3)
    b ← clamp(b_goal, 1, b_max, |feasible|)
    if near_boundary: b ← max(1, ⌊b/2⌋)                # soft-grad shrink (§6.7)

    for i in order[:b]:  u[i] ← clamp(u[i] − k·sign(g_i), 0, 255)   # k=1 byte
    C ← C ∪ order[:b]

anchor ← best_safe if exists else best_soft

# ---------- PHASE B: binary-search backward prune (optional) ----------  (§6.8)
# revert as many LEAST-salient changed coords as possible while staying in tier
sort C ascending by |g_i| → idx ;  keep ≥ 1 changed coord (so ‖δ‖∞ stays = q ≥ ε_lo)
binary-search largest r in [0, |C|−1] with ok_after_revert(idx[:r]):
    trial ← u with idx[:r] reset to clean bytes
    ok ⟺ ( tier=soft:  margin(trial) < 0 )  or  ( tier=safe:  margin(trial) ≤ −κ and envelope ≤ −κ )
apply the largest feasible revert

# ---------- PHASE C: finalize ----------  (§6.9)
re-evaluate anchor on the authoritative PNG round-trip (identity-skipped when grid-aligned)
return anchor.image if still flipped+in-band else x0 (clean ⇒ score 0)
```

`t_step` = measured cost of one forward+backward; all time guards are multiples of it so the miner
never starts work it cannot finish before `deadline = start + timeout − reserve`.

---

## 5. First-order model (the engine's core)

At the current point $x$ with margin $m=m(x)$ and gradient $g=\nabla_x m(x)$, a single byte step on
coordinate $i$ in the descent direction $\delta_i=-q\,\mathrm{sign}(g_i)$ gives, to first order,
$$
\Delta m \;\approx\; g^\top\delta \;=\; g_i\,\delta_i \;=\; -\,q\,|g_i| \;\le 0 .
$$
Flipping the top-$b$ coordinates by salience $|g_i|$ (descending order, prefix sum
$C_b=\sum_{j\le b}|g_{(j)}|$) predicts
$$
m\big(x+\delta^{(b)}\big)\ \approx\ m - q\,C_b .
$$
Greedy-by-$|g_i|$ is the **exact optimum of the linearized subproblem** "minimize $\#\{i:\delta_i\neq0\}$
s.t. predicted reduction $q\sum_{i}|g_i|\ge R$" — because for a fixed budget of nonzeros, the largest
$|g_i|$ give the most reduction per coordinate. The nonlinearity (curved boundary) is handled by
**re-linearizing every chunk** rather than trusting one linear solve.

---

## 6. Component math

### 6.1 CW margin & its gradient (`_margin_and_grad`)
$m=f_t-\max_{j\neq t}f_j$; $g=\nabla_x m$ via autograd. $m<0$ is exactly the validator's
misclassification test (subject to the numeric-transfer cushion below).

### 6.2 Re-linearization
The decision boundary $\{x:m(x)=0\}$ is curved. A one-shot linear solve over-counts the needed
coordinates by $\sim\!10\times$ near the boundary (empirically observed). Each chunk recomputes
$(m,g)$ at the **current perturbed point**, so the path tracks the boundary's curvature — a
matching-pursuit / SparseFool-style iteration.

### 6.3 Fractional chunk-sizing rule (`b_goal`)
Per chunk, the remaining margin to defeat is $R=m+\kappa$. We deliberately close only a fraction
$\alpha=\texttt{ALPHA}$ of it:
$$
b_{\text{goal}}=\min\Big\{b:\ q\,C_b\ \ge\ \alpha\,R\Big\},
\qquad b=\mathrm{clip}(b_{\text{goal}},\,1,\,b_{\max},\,|\text{feasible}|).
$$
Closing a fraction (not all) of $R$ keeps every step inside the regime where the linear model is
accurate, trading more forward passes for less over-shoot. **Convergence / step-count is one of the
analysis questions (§7).**

### 6.4 Per-chunk caps ($b_{\max}$)
- **Dimension cap:** $b\le \texttt{P\_STEP}\cdot n$ (don't change too large a fraction at once).
- **L2 / trust-region cap:** $b$ single-byte steps have $\|\delta_{\text{chunk}}\|_2=q\sqrt b$; requiring
  $q\sqrt b\le\texttt{RHO}$ gives $b\le(\texttt{RHO}/q)^2$. This bounds the move so the first-order
  Taylor expansion stays valid (a classic trust-region argument).
- $b_{\max}=\min$ of the two $\approx 30$ channels with defaults.

### 6.5 Byte-space construction (exactness)
All edits are integer bytes: $u=\mathrm{round}(255x_0)$, $u_i\leftarrow\mathrm{clamp}(u_i-k\,\mathrm{sign}(g_i),0,255)$,
$x=u/255$. Because each touched coordinate moves **exactly one byte and is never moved again**, the
candidate lies exactly on the $k/255$ grid $\Rightarrow$ the PNG encode/decode round-trip is a provable
identity (encoder uses `.round()`), and $\|\delta\|_\infty\equiv q$ holds by construction. This removes
float→uint8 ambiguity but does **not** address forward-pass numeric drift (that's §6.6).

### 6.6 Transfer cushion $\kappa$ and the TF32 envelope
The validator may run TF32 matmul/cudnn on/off; this perturbs logits and hence $m$. Two safeguards:
- **Margin cushion $\kappa$:** accept only if $m\le-\kappa$, so a flip has slack against drift.
- **TF32 envelope (`_margin_tf32on`):** evaluate $m$ with TF32 **off** ($m_{\text{off}}$) and **on**
  ($m_{\text{on}}$) and gate on the worst case
  $$
  \max(m_{\text{off}},\,m_{\text{on}})\ \le\ -\kappa .
  $$
  A flip surviving both endpoints survives the validator's unknown choice *by construction*, so $\kappa$
  shrinks from $0.01$ to a residual $0.005$ (only cross-GPU/library drift left). The off-margin
  pre-gates, so the extra TF32-on forward fires only when $m_{\text{off}}$ already clears the bar.

### 6.7 Soft-margin steering (optional, `PERTURB_SOFT_GRAD`)
Near the boundary the runner-up class identity $j^\star$ swaps between steps, making the hard gradient
$\nabla(f_t-f_{j^\star})$ jittery. The **ranking + step direction** can instead use a smoothed objective
over the top-$M$ competing logits $z_{(1)}\ge\dots\ge z_{(M)}$:
$$
m_{\text{soft}}\;=\;z_t\;-\;\tau\,\log\!\sum_{j=1}^{M}\exp\!\big(z_{(j)}/\tau\big)
\;=\;z_t-\tau\,\mathrm{logsumexp}\big(z_{(1:M)}/\tau\big).
$$
As $\tau\to0$, $m_{\text{soft}}\to$ hard margin; larger $\tau$ blends competitors for a steadier
descent direction. $(\tau,M)$ adapt to the gap $z_{(1)}-z_{(2)}$ and to boundary proximity. **Crucially,
the hard CW margin still makes every accept/break/flip decision** — soft only steers.

### 6.8 Binary-search backward prune monotonicity assumption
Order changed coordinates by ascending $|g_i|$ (least salient first) and let $\phi(r)=$ margin after
reverting the $r$ least-salient. The search assumes the predicate "$\phi(r)$ still in tier" is
**monotone** (true once false stays false), giving the largest revertible $r$ in $O(\log|S|)$ forwards.
$\phi$ is monotone *under the linear model* (each reverted coordinate adds back $+q|g_i|\ge0$ to the
margin), but the true $\phi$ is nonlinear. **Soundness of this monotonicity is an analysis question.**

### 6.9 Finalize invariant
The returned candidate is re-graded on the authoritative path; if it is not flipped + in-band it is
discarded and the **clean image** is returned (score 0 rather than a false claim). The miner never
emits an unverified candidate.

---

## 7. Assumptions to stress-test & open questions for the analyst

1. **Optimality gap of greedy + re-linearization vs. true min-$\ell_0$.** Greedy-by-$|g_i|$ is exact for
   the *linearized* subproblem each chunk, but the global min-$\ell_0$ problem is combinatorial. How far
   from optimal is the chunked path? Is there a submodularity / matroid structure to exploit, or a
   provable approximation ratio? (Margin-reduction is **not** submodular once we re-linearize.)
2. **Chunk-fraction $\alpha$ and convergence.** Does the "close $\alpha$ of remaining margin per chunk"
   rule have a guaranteed step count / contraction factor as a function of boundary curvature? Optimal
   $\alpha$ trading forward-pass count against over-shoot? Is a line-search / Armijo condition better
   than a fixed $\alpha$?
3. **Trust-region cap $\rho$.** Is $\|\delta_{\text{chunk}}\|_2\le\rho$ the right validity bound for the
   first-order model, given the resize Jacobian smooths the input? Could a curvature (Hessian-vector /
   Gauss–Newton) estimate justify larger, fewer chunks?
4. **Prune monotonicity (§6.8).** Is reverting least-salient-first provably monotone in the true margin,
   or can the binary search skip a valid sparser solution? Would a greedy one-at-a-time prune, or an
   $\ell_0$ re-solve on the final gradient, find a strictly sparser set?
5. **Sign/descent restriction.** We force $\sigma_i=-\mathrm{sign}(g_i)$. Could a small fraction of
   "wrong-sign" coordinates (interaction effects) reduce $|S|$? (Relevant to JSMA-style saliency that
   considers both signs.)
6. **Soft-margin objective.** Is logsumexp-over-top-$M$ the right smoothing? What $(\tau,M)$ schedule
   minimizes direction variance as $j^\star$ swaps? Relation to the smoothed-CW / margin-tempering
   literature.
7. **Convex relaxation alternative.** Would an $\ell_1$/elastic-net surrogate (EAD), a SparseFool
   linear-region solve, or an FMN-$\ell_0$ scheme provably beat the greedy heuristic on $|S|$ here,
   given $n\approx 12288$ and a tiny time budget (~1–6 s, tens of forwards)?
8. **Transfer model.** Is "worst case over TF32 {off,on} plus $\kappa$" a sufficient statistic for the
   numeric gap, or should drift be modeled as a distribution (e.g. bound $|m_{\text{validator}} -
   m_{\text{off}}|$) and $\kappa$ set from a confidence level rather than a constant?
9. **Diminishing-returns reality check.** Given score(1)=0.9523 vs score(369)=0.9396 (a 0.013 spread),
   where is the effort best spent — squeezing $|S|$, or hardening transfer so we never take a 0? Is
   there a closed-form expected-score objective $\mathbb E[\text{score}]=\Pr[\text{flip transfers}]\cdot
   \text{score}(|S|)$ that should be optimized directly instead of $|S|$ alone?

---

## 8. Quick reference — constants

```
n = 3·H·W,  H=W=64  ⇒ n = 12288         q = 1/255 = 3.9216e-3
ε_lo = 0.003   ε_hi = min(ε_syn, 0.03)   w_∞ = 0.7   w_r = 0.3
SPEED_WEIGHT = 0   PERTURBATION_WEIGHT = 1   SSIM ≥ 0.98   PSNR ≥ 38 dB
score(|S|) = 0.653023 + 0.3·(1 − 0.130719·√(|S|/n))²     # k=1, single-byte flips
ALPHA=0.30  P_STEP=0.0025  RHO=0.15  b_max≈30  κ∈{0.005 (envelope), 0.01}
```
