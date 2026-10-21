"""Frozen previous-round evidence for a teacher's complete hundred-chain population."""
from .io import read, write, better, eligible, value_sha


def stats(records):
    values = [r["pic50_on_rescore"] for r in records if eligible(r)]
    return {"attempts": len(records), "eligible": len(values), "failed": len(records) - len(values),
            "mean_pic50": sum(values) / len(values) if values else None,
            "best_pic50": max(values) if values else None}


def round_context(root, config, template, version, round_no):
    """Truncate resumed histories so later batches cannot see this round's outcomes."""
    if not 1 <= round_no <= config["rounds"]:
        raise ValueError("Population context requires a scheduled optimization round")
    folder = root / "versions" / version / template["teacher_id"]
    previous, incumbents, sources = [], [], []
    for batch in range(config["batches"]):
        chains = read(folder / f"batch_{batch:03d}/histories.json")
        if len(chains) != config["batch_size"] or [c["slot"] for c in chains] != list(range(config["batch_size"])):
            raise ValueError("Mean objective requires every original candidate slot")
        for c in chains:
            history = c["history"][:round_no]
            if [h["id"] for h in history] != [f"r{i:02d}" for i in range(round_no)]:
                raise ValueError("All five batches must finish the previous round before optimization advances")
            best = history[0]
            for h in history[1:]:
                if better(h["record"], best["record"]):
                    best = h
            previous.append(history[-1]["record"])
            incumbents.append(best["record"])
            sources.append({"batch": batch, "slot": c["slot"], "history_sha256": value_sha(history)})
    result = {"schema": 1, "teacher_id": template["teacher_id"], "version": version,
              "decision_round": round_no, "observed_round": round_no - 1,
              "priority": ["mean_pic50", "best_pic50"], "equal_chain_budget": True,
              "previous_unselected_round": stats(previous), "incumbent_population": stats(incumbents),
              "teacher_pic50": template["source_record"]["pic50_on_rescore"],
              "historical_baselines": {arm: stats(read(root / "baselines" / f"{template['target']}__{arm}.json"))
                                       for arm in ["unguided", "steer"]},
              "sources_sha256": value_sha(sources),
              "interpretation": "Mean over eligible scores; all 100 attempts and failures remain reported. "
                                "Never improve the mean by discarding a chain. Best is secondary. "
                                "Every slot uses this same pre-round population evidence."}
    path = folder / "round_contexts" / f"round_{round_no:02d}.json"
    if path.exists() and read(path) != result:
        raise ValueError("Frozen population context changed within a round")
    if not path.exists():
        write(path, result)
    return result
