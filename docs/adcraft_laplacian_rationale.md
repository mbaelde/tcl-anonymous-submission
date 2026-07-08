# Why we reimplemented the AdCraft environment

## Background

[Gomrokchi et al. (2023)](https://arxiv.org/abs/2306.11971) introduce the
**AdCraft** SEM bidding benchmark for constrained RL and use it in §7 to
evaluate their constrained-policy algorithms. We use the same benchmark in
§7.1 of the TCL paper to compare TCL against RCPO, Fixed, and HPRS.

The public reference implementation lives at
[Mikata-Project/adcraft](https://github.com/Mikata-Project/adcraft) and
ships a Rust extension (`adcraft` on PyPI). We initially used this package
via the wrapper `tcl/envs/adcraft_multiconstraint.py`.

## What we found

After an initial pilot (May 2026) produced degenerate results across all
four algorithms, we audited the upstream code against the paper
specification (§G.1 + Table 1, p. 21). We found **two independent
deviations** between the implementation and the paper:

### 1. Auction model — sigmoid impression rate instead of Laplacian critical bid

The paper §G.1 specifies an implicit auction model:

- Each auction draws a **critical bid** `c ~ |Laplace(loc_k, scale_k)|`.
- The agent wins when `b ≥ c` and **pays `c`** (second-price).
- Impression probability = `Pr(c ≤ b)` = Laplace CDF at `b`.

The upstream default (`sample_random_keywords` in `gymnasium_kw_utils.py`)
creates `ExplicitKeyword` objects instead, using a **sigmoid impression
rate** and `rust.cost_create` for the cost model. These are structurally
different from a Laplacian critical-bid auction.

### 2. `cost_create` — cost independent of bid, violates `cpc ≤ bid`

The `ExplicitKeyword` cost model delegates to `adcraft.rust.cost_create`.
The Rust source ([`src/lib.rs`](https://github.com/Mikata-Project/adcraft/blob/main/src/lib.rs)):

```rust
fn cost_create<'py>(py: Python<'py>, x: f64, n: usize) -> &'py PyArray1<f64> {
    let mut result_vec = Array::from_elem((n,), 4.4);  // filled with constant 4.4
    let x_sqrt = x.sqrt();
    let normal = Normal::new(0.0, 1e-10 + &x_sqrt / 6.0).unwrap();
    result_vec.iter_mut().for_each(|p| {
        *p = clamp(
            (&x_sqrt / 4.0 + *p / 2.0) + normal.sample(&mut thread_rng()),
            0.0,
            *p,   // clamp upper bound = 4.4, not the bid x
        )
    });
    result_vec.into_pyarray(py)
}
```

Bug: `p` is always 4.4 (the fill value, never updated), so `*p / 2.0 = 2.2`
is a fixed constant. Effective formula:
`clamp(sqrt(bid)/4 + 2.2 + N(0, sqrt(bid)/6), 0, 4.4)`.
The documented invariant `cpc ≤ bid` is violated for any bid < 4.4 —
the entire paper grid [0.01, 3.00]. Empirical measurements confirm the
analytical formula exactly:

```
bid=0.01 → mean cpc ≈ 2.22  (formula: sqrt(0.01)/4 + 2.2 = 2.225)
bid=0.10 → mean cpc ≈ 2.27  (formula: sqrt(0.10)/4 + 2.2 = 2.279)
bid=1.00 → mean cpc ≈ 2.45  (formula: sqrt(1)/4   + 2.2 = 2.450)
bid=10.0 → mean cpc ≈ 2.97  (formula: sqrt(10)/4  + 2.2 = 2.990)
```

This explains the pilot results: with budget=100 and mean cpc ≈ 2.5,
only ~40 buyside clicks fit in the budget per step — spending patterns
are driven by `cost_create`'s internal constant, not by the agent's bid.

No GitHub issue or public discussion acknowledges this bug (verified
2026-05-19: 0 external issues, 2 inactive forks).

**Note on `sctr`:** `sample_random_keywords` samples
`sctr ~ Beta(5, 2)` (mean ≈ 0.714), which **matches** Table 1 of the
paper. This is not a deviation.

## Upstream status

As of August 2023 the upstream repository has been **unmaintained**:

- Last commit: 2023-08-22 (PR #17)
- Issues: 0 opened or closed since creation
- 17 PRs, all merged between May–August 2023; none touch the cost
  distribution, `cost_create`, or `sctr` Beta parameters
- 2 forks, both inactive for 2+ years

The deviations above are therefore the **official implementation**, not a
local drift. A patch PR would have no realistic path to merge.

## Decision: pure-Python reimplementation

We chose to **rewrite the environment from scratch in pure Python**,
implementing §G.1 + Table 1 directly:

- `tcl/envs/adcraft_laplacian_sim.py` — `BiddingSimulationLaplacian`
- `tcl/envs/adcraft_laplacian.py` — `MultiConstraintAdCraftLaplacian`

This gives us:

- **Correctness**: critical bid `c ~ |Laplace(loc_k, scale_k)|`, second-price
  mechanic, `sctr ~ Beta(5, 2)`, reward `~ TruncNormal(μ_R, σ_R, min=0.01)`.
- **No Rust dependency**: installable anywhere with `uv sync`.
- **Auditability**: every distribution draw is a traceable NumPy call.
- **Flexibility**: `pricing_mode="first"` variant for first-price settings
  (e.g. Anonymous Institution production RTB auction).

The legacy wrapper `tcl/envs/adcraft_multiconstraint.py` is retained for
backward compatibility with pre-2026-05-19 experiment artefacts.

The justification text for §7.1 of the paper reads:

> We re-implement the AdCraft environment from §G.1 in pure Python, as the
> public codebase (`Mikata-Project/adcraft`) has been unmaintained since
> August 2023 and deviates from the paper specification on two independent
> points: (1) the default keyword sampler uses `ExplicitKeyword` with a
> sigmoid impression model rather than the `ImplicitKeyword` Laplacian
> second-price auction of §G.1; (2) the compiled Rust function `cost_create`
> contains a bug (hardcoded constant 4.4) that makes cost nearly independent
> of the submitted bid, violating the documented `cpc ≤ bid` invariant for
> any bid below 4.4. Our reimplementation is faithful to §G.1 + Table 1 and
> ships no compiled extension.
