"""
End-to-end feasibility check for the research planner.

This is not a replacement for the real train.py/val_bpb loop. It is a tiny CPU
proxy that proves the planner architecture is wired correctly:

candidate ideas -> heuristic scoring -> archive/diversity selection ->
small training runs -> result log -> archive update.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_planner import (
    HeuristicRewardModel,
    Idea,
    IdeaArchive,
    ScoredIdea,
    default_literature,
    format_score_table,
    select_for_execution,
)


@dataclass(frozen=True)
class ToyConfig:
    idea: Idea
    hidden_dim: int = 64
    lr: float = 3e-3
    dropout: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 64
    optimizer: str = "adamw"
    cosine_warmdown: bool = False


@dataclass(frozen=True)
class TrainResult:
    val_loss: float
    train_seconds: float
    num_params: int


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31)


def candidate_experiments() -> dict[str, ToyConfig]:
    experiments = [
        ToyConfig(
            idea=Idea(
                idea_id="lower_lr_cosine",
                title="Lower LR with cosine warmdown",
                description="Reduce optimizer learning rate and add a late cosine warmdown to improve endpoint validation loss.",
                niche="lr_schedule",
                patch_plan="In train.py, lower Adam/Muon LR or increase warmdown smoothness while keeping the fixed time budget.",
                tags=("lr", "warmdown", "stability"),
                prior=0.72,
                risk=0.22,
                effort=0.25,
            ),
            lr=2e-3,
            cosine_warmdown=True,
        ),
        ToyConfig(
            idea=Idea(
                idea_id="wider_hidden",
                title="Wider hidden dimension",
                description="Increase model width for more capacity while watching throughput and memory.",
                niche="architecture",
                patch_plan="Increase ASPECT_RATIO or effective n_embd in train.py and compare fixed-time val_bpb.",
                tags=("width", "capacity", "memory"),
                prior=0.66,
                risk=0.42,
                effort=0.45,
            ),
            hidden_dim=96,
        ),
        ToyConfig(
            idea=Idea(
                idea_id="dropout_regularization",
                title="Add small dropout regularization",
                description="Use light dropout to reduce overfitting in small proxy training.",
                niche="regularization",
                patch_plan="For real train.py this is a risky idea because dropout may hurt throughput; validate only if other knobs saturate.",
                tags=("dropout", "regularization"),
                prior=0.42,
                risk=0.55,
                effort=0.35,
            ),
            dropout=0.08,
        ),
        ToyConfig(
            idea=Idea(
                idea_id="adamw_weight_decay",
                title="Increase AdamW weight decay",
                description="Add moderate weight decay to non-embedding parameters to improve generalization.",
                niche="optimizer",
                patch_plan="Tune WEIGHT_DECAY and compare against the baseline Muon/AdamW split.",
                tags=("adamw", "weight_decay", "optimizer"),
                prior=0.58,
                risk=0.30,
                effort=0.25,
            ),
            weight_decay=0.03,
        ),
        ToyConfig(
            idea=Idea(
                idea_id="smaller_faster_model",
                title="Smaller faster model",
                description="Reduce hidden size to trade capacity for more optimizer steps within a fixed budget.",
                niche="efficiency",
                patch_plan="Lower DEPTH or ASPECT_RATIO in train.py and measure whether extra steps offset reduced capacity.",
                tags=("throughput", "small_model", "fixed_budget"),
                prior=0.54,
                risk=0.33,
                effort=0.30,
            ),
            hidden_dim=40,
            lr=3.5e-3,
        ),
        ToyConfig(
            idea=Idea(
                idea_id="sgd_momentum",
                title="SGD momentum optimizer",
                description="Try a simpler momentum optimizer as an optimizer-family exploration.",
                niche="optimizer",
                patch_plan="This would be a high-risk real train.py change because Muon is specialized for matrix updates.",
                tags=("sgd", "momentum", "optimizer"),
                prior=0.28,
                risk=0.70,
                effort=0.40,
            ),
            optimizer="sgd",
            lr=5e-2,
        ),
    ]
    return {experiment.idea.idea_id: experiment for experiment in experiments}


def baseline_experiment() -> ToyConfig:
    return ToyConfig(
        idea=Idea(
            idea_id="baseline",
            title="Baseline tiny MLP",
            description="Baseline proxy run used only to compute delta for selected ideas.",
            niche="baseline",
            prior=0.5,
            risk=0.1,
            effort=0.1,
        )
    )


def make_dataset(seed: int, *, n_train: int = 2048, n_val: int = 1024) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    input_dim = 20
    teacher_dim = 14
    num_classes = 5

    teacher_w1 = torch.randn(input_dim, teacher_dim, generator=generator)
    teacher_w2 = torch.randn(teacher_dim, num_classes, generator=generator)

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, input_dim, generator=generator)
        logits = torch.tanh(x @ teacher_w1) @ teacher_w2
        logits = logits + 0.05 * torch.randn(n, num_classes, generator=generator)
        y = logits.argmax(dim=1)
        return x, y

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    return x_train, y_train, x_val, y_val


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_toy_config(config: ToyConfig, *, steps: int, seed: int) -> TrainResult:
    torch.manual_seed(stable_seed(seed, config.idea.idea_id))
    x_train, y_train, x_val, y_val = make_dataset(seed)
    model = TinyMLP(x_train.size(1), config.hidden_dim, int(y_train.max().item()) + 1, config.dropout)

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {config.optimizer}")

    start = time.time()
    for step in range(steps):
        if config.cosine_warmdown:
            progress = step / max(1, steps - 1)
            lr_multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = config.lr * lr_multiplier

        idx = torch.randint(0, x_train.size(0), (config.batch_size,))
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    train_seconds = time.time() - start
    model.eval()
    with torch.no_grad():
        val_loss = F.cross_entropy(model(x_val), y_val).item()
    num_params = sum(param.numel() for param in model.parameters())
    return TrainResult(val_loss=val_loss, train_seconds=train_seconds, num_params=num_params)


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "idea_id",
                "niche",
                "predicted_score",
                "toy_val_loss",
                "delta_vs_baseline",
                "status",
                "seconds",
                "params",
                "title",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    archive_path = out_dir / "idea_archive.jsonl"
    results_path = out_dir / "toy_results.tsv"
    if args.fresh and archive_path.exists():
        archive_path.unlink()

    archive = IdeaArchive.from_jsonl(archive_path)
    candidates = candidate_experiments()
    reward_model = HeuristicRewardModel(default_literature(), archive)
    scored = reward_model.score_many([experiment.idea for experiment in candidates.values()])
    selected = select_for_execution(scored, archive, k=args.k)

    print("Candidate scores:")
    print(format_score_table(scored))
    print()
    print("Selected for toy execution:")
    if selected:
        print(format_score_table(selected))
    else:
        print("No non-redundant ideas available. Use --fresh to clear the toy archive.")
        return

    baseline = baseline_experiment()
    baseline_result = train_toy_config(baseline, steps=args.steps, seed=args.seed)
    print()
    print(f"Baseline toy_val_loss: {baseline_result.val_loss:.6f}")

    rows: list[dict[str, str]] = [
        {
            "idea_id": baseline.idea.idea_id,
            "niche": baseline.idea.niche,
            "predicted_score": "",
            "toy_val_loss": f"{baseline_result.val_loss:.6f}",
            "delta_vs_baseline": "0.000000",
            "status": "baseline",
            "seconds": f"{baseline_result.train_seconds:.3f}",
            "params": str(baseline_result.num_params),
            "title": baseline.idea.title,
        }
    ]

    for scored_idea in selected:
        config = candidates[scored_idea.idea.idea_id]
        result = train_toy_config(config, steps=args.steps, seed=args.seed)
        delta = baseline_result.val_loss - result.val_loss
        status = "keep" if delta > 0 else "discard"
        archive.add_result(
            scored_idea,
            status=status,
            metric_name="toy_val_loss",
            metric_value=result.val_loss,
            metric_delta=delta,
            notes=f"steps={args.steps}; params={result.num_params}; seconds={result.train_seconds:.3f}",
        )
        rows.append(
            {
                "idea_id": config.idea.idea_id,
                "niche": config.idea.niche,
                "predicted_score": f"{scored_idea.expected_score:.6f}",
                "toy_val_loss": f"{result.val_loss:.6f}",
                "delta_vs_baseline": f"{delta:.6f}",
                "status": status,
                "seconds": f"{result.train_seconds:.3f}",
                "params": str(result.num_params),
                "title": config.idea.title,
            }
        )
        print(
            f"{config.idea.idea_id}: toy_val_loss={result.val_loss:.6f} "
            f"delta={delta:+.6f} status={status}"
        )

    write_results(results_path, rows)
    archive.save_jsonl(archive_path)
    print()
    print(f"Wrote {results_path}")
    print(f"Wrote {archive_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end planner validation.")
    parser.add_argument("--out-dir", default="planner_runs/toy", help="Directory for toy archive and results.")
    parser.add_argument("--k", type=int, default=3, help="Number of candidate ideas to execute.")
    parser.add_argument("--steps", type=int, default=80, help="Training steps per toy experiment.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for the proxy task.")
    parser.add_argument("--fresh", action="store_true", help="Clear the toy archive before selecting ideas.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
