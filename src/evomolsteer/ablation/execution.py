"""Execution failures are operational outcomes, not causal Skill effects."""


class ExecutionBudgetExceeded(RuntimeError):
    """Stop the whole queue rather than misclassifying a shared budget limit."""


class ModuleExecutionError(ValueError):
    def __init__(self, case, stage, error, *, role=None, chain=None):
        self.case = case
        self.stage = stage
        self.original_error_type = type(error).__name__
        self.role = role
        self.chain = chain
        self.diagnostics = getattr(error, "diagnostics", None)
        super().__init__(str(error))


def failure_record(module, phase, active, error):
    """Retain the instruction; exclude an incomplete comparison from pruning."""
    return {
        "module": module, "phase": phase, "status": "SKIPPED_EXECUTION_ERROR",
        "case": getattr(error, "case", None),
        "stage": getattr(error, "stage", "module_comparison"),
        "role": getattr(error, "role", None), "chain": getattr(error, "chain", None),
        "error_type": getattr(error, "original_error_type", type(error).__name__),
        "error": str(error)[:2000], "diagnostics": getattr(error, "diagnostics", None),
        "active_before": list(active), "selected_active": list(active),
        "instruction_removed": False, "performance_comparison_valid": False,
        "skills_ablation_execution_abnormality_observed": True,
        "ablation_caused_the_error": "UNRESOLVED",
        "causal_claim_supported": False,
        "interpretation": "This condition failed execution; it does not establish that removing this instruction caused a Skill or performance failure.",
        "original_outputs_preserved": True,
    }
