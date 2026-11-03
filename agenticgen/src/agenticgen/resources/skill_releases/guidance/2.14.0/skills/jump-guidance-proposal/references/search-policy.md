# Search-policy references

This release adapts branching, measured feedback, forward expansion, and backtracking to a finite guidance-parameter search. It does not implement Monte Carlo Tree Search, language-model value scoring, or intermediate molecular-state rollouts.

- Yao et al., [Tree of Thoughts](https://arxiv.org/abs/2305.10601): alternative paths, lookahead, and backtracking. [Official prompts and implementation](https://github.com/princeton-nlp/tree-of-thought-llm).
- Zhou et al., [Language Agent Tree Search](https://arxiv.org/abs/2310.04406): environment feedback for adaptive exploration. [Official implementation](https://github.com/andyz245/LanguageAgentTreeSearch).

For this application, a node is a complete measured six-scale schedule, an edge records a real parent and a new allocation, and the evaluator supplies the condition-aligned mean. The search focus is a decision aid, not a simulation checkpoint. Five proposed configurations are evaluated in full; no language-model selection stage narrows them.

Example: two probes increasing B1 by one and two grid positions have the same early signed pattern and count as one direction. Increasing B1 and decreasing B1 are different directions. Changing B3 alone and changing B4 alone are different directions. A new suffix attached to an unchanged historical prefix is a continuation, not a fresh early perturbation.
