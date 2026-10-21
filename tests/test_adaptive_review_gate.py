import copy
import subprocess

import pytest

from evomolsteer.adaptive_schedule import review_gate
from evomolsteer.adaptive_schedule.io import defaults, read, write, sha


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    root, repo = tmp_path / "run", tmp_path / "repo"
    repo.mkdir()
    config = defaults()
    config.update(batches=1, batch_size=2, rounds=2, retention={"mode": "all"})
    teachers = [{"teacher_id": "teacher", "target": "target"}]
    write(root / "teachers.json", teachers)
    chains = [{"slot": s, "best_id": "r02", "history": [{"id": f"r{r:02d}"} for r in range(3)]}
              for s in range(2)]
    write(root / "versions/v1/teacher/batch_000/histories.json", chains)
    write(root / "versions/v1/teacher/batch_000/COMPLETE.json", {"chains": 2, "rounds": 2})
    write(root / "versions/v1/COMPLETE.json", {"decisions": 4})
    write(root / "FINALIZED_v1.json", {"review_decisions": 4})
    write(root / "versions/v1/binding.json", {"skill_sha256": "previous-skill"})
    rows = []
    for s in range(2):
        for r in range(1, 3):
            evidence = f"evidence/s{s}_r{r}.json"
            write(root / evidence, {"evidence": {"r00:t050": {"local_loss": 2}}})
            rows.append({"teacher": "teacher", "batch": 0, "slot": s, "id": f"r{r:02d}",
                         "prompt_evidence": evidence, "prompt_evidence_sha256": sha(root / evidence)})
    shard = "review/v1/shard_0000.json"
    write(root / shard, rows)
    write(root / "review/v1/manifest.json", {"version": "v1", "decisions": 4,
          "shards": [{"path": shard, "sha256": sha(root / shard), "decisions": 4}]})
    write(repo / "reviews/shard_0000.json", {"shard_sha256": sha(root / shard), "decisions": [
        dict({k: row[k] for k in ["teacher", "batch", "slot", "id"]},
             assessment="Evidence supports a smaller early perturbation; clipping remains a counterexample.",
             evidence_ids=["r00:t050"]) for row in rows]})
    (repo / "SKILL.md").write_text("Revised general skill\n", encoding="utf-8")
    (repo / "review.md").write_text("All four decisions reviewed; distinguish geometry from score.\n", encoding="utf-8")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "review")
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(review_gate, "repository", lambda: repo)
    monkeypatch.setattr(review_gate, "commit", lambda: revision)
    approval = {"schema": 1, "manifest_sha256": sha(root / "review/v1/manifest.json"),
        "finalized_sha256": sha(root / "FINALIZED_v1.json"),
        "v1_binding_sha256": sha(root / "versions/v1/binding.json"), "review_commit": revision,
        "skill_path": "SKILL.md", "skill_sha256": sha(repo / "SKILL.md"),
        "review_document": "review.md", "review_document_sha256": sha(repo / "review.md"),
        "coverage": [{"path": shard, "sha256": sha(root / shard),
                      "review_file": "reviews/shard_0000.json", "review_sha256": sha(repo / "reviews/shard_0000.json")}]}
    write(root / "versions/v2/review_approval.json", approval)
    return root, repo, config, approval


def test_committed_full_review_can_start_v2(reviewed):
    root, repo, config, _ = reviewed
    result = review_gate.validate_review(root, config, repo / "SKILL.md")
    assert result["decisions"] == 4 and result["shards"] == 1


@pytest.mark.parametrize("change", ["empty_approval", "missing_coverage", "duplicate_coverage", "wrong_commit",
                                    "edited_skill", "edited_review", "edited_evidence", "incomplete_history",
                                    "missing_finalized", "edited_shard", "escape"])
def test_incomplete_or_stale_approval_is_rejected(reviewed, change):
    root, repo, config, approval = reviewed
    approval = copy.deepcopy(approval)
    if change == "empty_approval": approval = {}
    elif change == "missing_coverage": approval["coverage"] = []
    elif change == "duplicate_coverage": approval["coverage"] *= 2
    elif change == "wrong_commit": approval["review_commit"] = "a" * 40
    elif change == "edited_skill": (repo / "SKILL.md").write_text("uncommitted")
    elif change == "edited_review": (repo / "reviews/shard_0000.json").write_text("{}")
    elif change == "edited_evidence": write(root / "evidence/s0_r1.json", {"evidence": {}})
    elif change == "incomplete_history":
        p = root / "versions/v1/teacher/batch_000/histories.json"
        chains = read(p); chains[0]["history"].pop(); write(p, chains)
    elif change == "missing_finalized": (root / "FINALIZED_v1.json").unlink()
    elif change == "edited_shard": write(root / "review/v1/shard_0000.json", [])
    elif change == "escape": approval["skill_path"] = "../SKILL.md"
    write(root / "versions/v2/review_approval.json", approval)
    with pytest.raises((ValueError, OSError)):
        review_gate.validate_review(root, config, repo / "SKILL.md")
