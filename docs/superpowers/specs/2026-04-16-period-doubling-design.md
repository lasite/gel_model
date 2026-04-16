# Design: biasing the current gel model toward period-doubling

## Goal

Increase the chance that the existing 1D slab `J-u-theta` model develops a stable route toward `P2 -> P4 -> chaos` without changing the governing state variables or adding a new recovery field.

The target is not just to obtain a single isolated `P2` point. The target is to make period-doubling more structurally accessible by widening the region where cross-cycle memory survives and by moving the existing `P1` limit cycle closer to a flip instability.

## Scope

In scope:

- tune only existing parameters and existing constitutive function shapes
- keep the current PDE structure and `J-u-theta` state vector
- evaluate changes primarily along the fixed-`Bi_c = 0.70` Da continuation line
- use the `x=L` probe for all oscillation-type decisions

Out of scope:

- adding a new state variable
- introducing a new reaction mechanism
- redesigning the thermal or chemical potential model

## Design summary

The model should be pushed toward period-doubling by strengthening **recovery mismatch across cycles**, not by simply making the positive thermal feedback larger.

The main design principle is:

1. make transport/accessibility recover more slowly after collapse
2. preserve enough re-supply that the oscillation does not terminate
3. only then use small thermal-feedback adjustments to push the resulting limit cycle toward flip loss of stability

This keeps the modification physically interpretable: the gel remains the same thermo-chemo-poroelastic oscillator, but the collapsed state leaves a longer-lived memory into the next cycle.

## Candidate approaches considered

### Approach A: recovery-limited tuning (recommended)

Tune existing transport/accessibility controls:

- `m_diff`
- `m_mob`
- `m_act`
- small coordinated adjustments in `Bi_T` and optionally `Bi_c`

Why this is preferred:

- it stays closest to the current physical interpretation
- it directly strengthens the slow negative feedback arm
- it is the most natural way to create cycle-to-cycle alternation in a relaxation-type oscillator

Main risk:

- if recovery is made too slow, the oscillation dies into steady or into a single strongly clipped P1 branch

### Approach B: sharper constitutive gating

Keep the same variables and laws but make existing `J`-dependent transport/accessibility curves effectively more switch-like.

Why it might work:

- sharper collapse gating can move the P1 branch closer to a flip instability

Why it is not the first choice:

- it is still within scope, but it starts to make the constitutive behavior look more artificially tuned

### Approach C: stronger thermal forcing

Primarily tune `Gamma_A` and `S_chi` upward while balancing with `Bi_T`.

Why it was not chosen as the primary route:

- this often hardens the existing P1 relaxation oscillation instead of generating robust alternation
- it increases stiffness before it reliably creates a wide period-doubling window

## Chosen design

Use **Approach A** as the main path, with **small amounts of Approach B/C only after A creates a near-P2 region**.

The tuning order matters:

### Stage 1: transport recovery

First increase transport-memory effects:

1. raise `m_diff`
2. then raise `m_mob`

Expected effect:

- after collapse, reactant penetration and swelling recovery become less complete by the time the next ignition cycle begins
- the system retains a stronger cross-cycle memory

### Stage 2: accessibility gating

After recovery lag is stronger, increase `m_act`.

Expected effect:

- the distinction between “reactive swollen state” and “collapsed inactive state” becomes sharper
- small differences in recovery state can produce larger differences in next-cycle ignition timing or amplitude

### Stage 3: small thermal push

Only after the system shows a broader near-P2 region:

- modestly raise `Gamma_A`, or
- modestly raise `S_chi`, or
- slightly lower `Bi_T`

Expected effect:

- the already memory-rich P1 branch is nudged closer to flip instability
- thermal forcing amplifies an existing two-cycle tendency instead of trying to create it from nothing

## Experimental protocol

The first evaluation path is a single high-quality bifurcation line:

- fixed `Bi_c = 0.70`
- sweep `Da`
- use forward and backward continuation

For each tuned parameter set:

1. run the fixed-`Bi_c` Da continuation
2. classify each point using the `x=L` trajectory
3. record:
   - label (`steady / P1 / P2 / Pn / chaos`)
   - `T_mean`
   - `T_ratio`
   - permutation entropy
   - forward/backward agreement or disagreement
4. inspect whether the P2 region widens, shifts, disappears, or turns into higher-order structure

## Success criteria

### Level 1 success

A clearly wider and more stable `P2` island along the fixed-`Bi_c = 0.70` Da line.

### Level 2 success

Evidence of stable higher-order doubling behavior:

- persistent `P4`, or
- a sequence of higher-period windows adjacent to the `P2` region

### Level 3 success

Candidate chaotic behavior that survives refinement checks:

- remains after longer integration
- remains when checked at higher resolution (`N=60`)
- is not merely transient or classification noise

## Failure criteria

The design is considered unsuccessful if one of the following dominates:

- the model remains essentially all-`P1`
- the P2 window collapses rather than broadens
- the system is pushed mostly into steady termination
- apparent higher-period behavior disappears at `N=60`

## Verification strategy

Use a two-stage resolution policy:

1. `N=40` for fast screening
2. `N=60` for any candidate `P2/P4/chaos` region

The authoritative observable remains the `x=L` node. Interior MMO-like subpeaks are not to be treated as period-doubling evidence.

## Expected outcome

If the design works, the model should move from:

- a dominant robust `P1` branch with only a narrow `P2` island

to:

- a broader `P2` region, then
- a neighborhood with higher-period structure or chaos candidates, driven by incomplete inter-cycle recovery rather than by an unrelated new mechanism.
