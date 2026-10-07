Decision: **test_formula**. Set only `reward_view=endpoint_path_value`; compare with the completed terminal-source `endpoint_pointcloud` parent. Freeze source, controller, dose, window `[0.0,0.5]`, scalars and evaluation. History, geometry projection and contrast amplitude remain disabled.

For fixed detached neighbors, assignments and prior, the registered scalar is

`R = tau * [logsumexp(log(b)+beta*(u-mean(u))-C/tau) - logsumexp(log(b)-C/tau)]`.

Its endpoint gradient is `-sum_j (p_u[j]-p_0[j]) * grad C_j`, where `p_u` and `p_0` are the utility-weighted and background kernel softmax distributions. Subtracting the background removes their common geometric attraction. Constant utilities make both distributions identical, giving zero reward and zero gradient; one neighbor also gives zero gradient. This is a testable change in the force, not evidence of an affinity gain.

The coordinate observable is matched FLOWR endpoint displacement: `q` is mean squared displacement and `C=delta^2*(sqrt(1+q/delta^2)-1)/bandwidth`. Hungarian correspondence and neighbor selection use a detached anchor. Differentiate this cost to the endpoint and continue through the required real FLOWR endpoint VJP to the current state. The reference supplies detached observed terminal Steer predicted-affinity labels and donor/ancestor mass; this module consumes mass without verifying its construction. Existing scalars are frozen experimental choices. Neither the labels nor the gradient identify native-future probabilities, causal hotspots or interaction energies.

Counterevidence is material: source replacement alone gave zero valid elites in 50 outputs, only +0.023600 paired mean pIC50 versus gradient, weaker validity/PB and worse strain p90. The one-batch result cannot establish reproducibility. The packet reports five passing unit tests; inspected formula tests cover flat-label and single-neighbor plateaus, a symmetric label swap and fixed-anchor finite differences. Actual inference, integrated endpoint VJP, regional causal evidence and native-continuation evidence remain missing.

Test on the supplied paired batch30/seed42/n=50 panel with identical terminal decoding/rescore, scorer clock 0.9999 and elite threshold 8.258901977539063. The primary prediction is higher valid elite yield per all outputs and more unique elite graphs. Also report complete score distributions, validity/PB, strain median/p90 and convergence counts. A mean gain or duplicate-only gain does not establish this prediction. A nonpositive elite-yield difference fails this screen; zero events in both arms remains inconclusive for a small population effect. Independent paired batches are required for a general benefit claim.

Failure modes: collapsed local contrast; discrete assignment/neighbor switches; sparse early support; selected/censored/clone-dependent labels; and losing a useful density pull, with possible validity/PB or strain regressions. The derivative is conditional on frozen support, and scalar ascent alone does not establish molecular quality.

Request bindings (input and instruction hashes independently matched):

- `input_sha256`: `61eb8f64be95a5ae01af9b241bf9b29b785a207901f022d2dbfed7cb3e6960a5`
- `instruction_sha256`: `f03ef6560e10c1014f491889f64eaa41adbef6920f6a0476d9288edddfeff9c2`
- `skill_sha256` supplied in request: `6ea6aecf91f06e57ae443a4960ac879a4af1fe0a7b9a2d3f2fbddfe2cc5bc46c` (not independently recomputed).
