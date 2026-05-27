"""
Lightweight research planning layer for autoresearch.

This file implements a concrete, testable version of the two design sketches:

1. Score candidate ideas before execution.
2. Keep an archive organized by niches and select a diverse execution batch.

The planner is intentionally independent from train.py. The real reward still
comes from running train.py and measuring val_bpb; this layer only decides which
ideas deserve scarce GPU time.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _term_freq(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vec.values()))


def text_similarity(left: str, right: str) -> float:
    """Cosine similarity over simple bag-of-words features."""
    left_vec = _term_freq(left)
    right_vec = _term_freq(right)
    left_norm = _norm(left_vec)
    right_norm = _norm(right_vec)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left_vec) > len(right_vec):
        left_vec, right_vec = right_vec, left_vec
    dot = sum(value * right_vec.get(token, 0.0) for token, value in left_vec.items())
    return dot / (left_norm * right_norm)


class SimilarityIndex:
    """Tiny in-memory text index used for literature and archive similarity."""

    def __init__(self, documents: Iterable[str]):
        self._vectors = [_term_freq(doc) for doc in documents if doc.strip()]
        self._norms = [_norm(vec) for vec in self._vectors]

    def max_similarity(self, text: str) -> float:
        query_vec = _term_freq(text)
        query_norm = _norm(query_vec)
        if query_norm == 0.0 or not self._vectors:
            return 0.0

        best = 0.0
        for doc_vec, doc_norm in zip(self._vectors, self._norms):
            if doc_norm == 0.0:
                continue
            if len(query_vec) <= len(doc_vec):
                dot = sum(value * doc_vec.get(token, 0.0) for token, value in query_vec.items())
            else:
                dot = sum(value * query_vec.get(token, 0.0) for token, value in doc_vec.items())
            best = max(best, dot / (query_norm * doc_norm))
        return best


@dataclass(frozen=True)
class LiteratureItem:
    title: str
    summary: str
    tags: tuple[str, ...] = ()

    def text(self) -> str:
        return " ".join((self.title, self.summary, " ".join(self.tags)))


@dataclass(frozen=True)
class Idea:
    idea_id: str
    title: str
    description: str
    niche: str
    patch_plan: str = ""
    tags: tuple[str, ...] = ()
    prior: float = 0.5
    risk: float = 0.35
    effort: float = 0.4

    def text(self) -> str:
        return " ".join(
            (
                self.idea_id,
                self.title,
                self.description,
                self.niche,
                self.patch_plan,
                " ".join(self.tags),
            )
        )


@dataclass(frozen=True)
class ScoredIdea:
    idea: Idea
    literature_similarity: float
    archive_similarity: float
    novelty: float
    evidence: float
    redundancy: float
    usefulness: float
    feasibility: float
    simplicity: float
    risk_penalty: float
    expected_score: float


@dataclass(frozen=True)
class ArchiveEntry:
    idea_id: str
    title: str
    niche: str
    text: str
    predicted_score: float
    status: str
    metric_name: str = ""
    metric_value: float | None = None
    metric_delta: float | None = None
    notes: str = ""


class IdeaArchive:
    """Structured memory of ideas that were already selected or executed."""

    def __init__(self, entries: Sequence[ArchiveEntry] = ()):
        self.entries = list(entries)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "IdeaArchive":
        archive_path = Path(path)
        if not archive_path.exists():
            return cls()

        entries: list[ArchiveEntry] = []
        for line in archive_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(ArchiveEntry(**json.loads(line)))
        return cls(entries)

    def save_jsonl(self, path: str | Path) -> None:
        archive_path = Path(path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.dumps(asdict(entry), sort_keys=True) for entry in self.entries]
        archive_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def text_index(self) -> SimilarityIndex:
        return SimilarityIndex(entry.text for entry in self.entries)

    def max_similarity(self, idea: Idea) -> float:
        return self.text_index().max_similarity(idea.text())

    def idea_ids(self) -> set[str]:
        return {entry.idea_id for entry in self.entries}

    def add_result(
        self,
        scored: ScoredIdea,
        *,
        status: str,
        metric_name: str = "",
        metric_value: float | None = None,
        metric_delta: float | None = None,
        notes: str = "",
    ) -> None:
        idea = scored.idea
        self.entries.append(
            ArchiveEntry(
                idea_id=idea.idea_id,
                title=idea.title,
                niche=idea.niche,
                text=idea.text(),
                predicted_score=scored.expected_score,
                status=status,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_delta=metric_delta,
                notes=notes,
            )
        )


@dataclass(frozen=True)
class RewardWeights:
    usefulness: float = 0.32
    feasibility: float = 0.23
    evidence: float = 0.15
    novelty: float = 0.10
    simplicity: float = 0.10
    redundancy: float = 0.25
    risk: float = 0.10


class HeuristicRewardModel:
    """
    Bootstrap reward model for idea ranking.

    This is deliberately heuristic. Once enough real experiments exist, the same
    feature vector can be replaced by a learned model trained on val_bpb deltas.
    """

    def __init__(
        self,
        literature: Sequence[LiteratureItem],
        archive: IdeaArchive,
        weights: RewardWeights = RewardWeights(),
    ):
        self._literature = list(literature)
        self._literature_index = SimilarityIndex(item.text() for item in literature)
        self._archive = archive
        self._archive_index = archive.text_index()
        self._weights = weights

    def score(self, idea: Idea) -> ScoredIdea:
        literature_similarity = self._literature_index.max_similarity(idea.text())
        archive_similarity = self._archive_index.max_similarity(idea.text())

        novelty = 1.0 - literature_similarity
        evidence = literature_similarity if self._literature else 0.5
        redundancy = archive_similarity
        usefulness = _clamp(idea.prior)
        feasibility = _clamp(1.0 - 0.65 * idea.risk - 0.35 * idea.effort)
        simplicity = _clamp(1.0 - idea.effort)
        risk_penalty = _clamp(idea.risk)

        w = self._weights
        expected_score = (
            w.usefulness * usefulness
            + w.feasibility * feasibility
            + w.evidence * evidence
            + w.novelty * novelty
            + w.simplicity * simplicity
            - w.redundancy * redundancy
            - w.risk * risk_penalty
        )

        return ScoredIdea(
            idea=idea,
            literature_similarity=round(literature_similarity, 6),
            archive_similarity=round(archive_similarity, 6),
            novelty=round(novelty, 6),
            evidence=round(evidence, 6),
            redundancy=round(redundancy, 6),
            usefulness=round(usefulness, 6),
            feasibility=round(feasibility, 6),
            simplicity=round(simplicity, 6),
            risk_penalty=round(risk_penalty, 6),
            expected_score=round(expected_score, 6),
        )

    def score_many(self, ideas: Sequence[Idea]) -> list[ScoredIdea]:
        return sorted((self.score(idea) for idea in ideas), key=lambda item: item.expected_score, reverse=True)


def _marginal_diversity(candidate: ScoredIdea, selected: Sequence[ScoredIdea]) -> float:
    if not selected:
        return 1.0
    similarities = [text_similarity(candidate.idea.text(), item.idea.text()) for item in selected]
    return 1.0 - max(similarities)


def _insert_nonredundant(
    cell: list[ScoredIdea],
    candidate: ScoredIdea,
    redundancy_threshold: float,
) -> None:
    for idx, existing in enumerate(cell):
        similarity = text_similarity(candidate.idea.text(), existing.idea.text())
        if similarity >= redundancy_threshold:
            if candidate.expected_score > existing.expected_score:
                cell[idx] = candidate
            return
    cell.append(candidate)


def select_for_execution(
    scored_ideas: Sequence[ScoredIdea],
    archive: IdeaArchive,
    *,
    k: int,
    redundancy_threshold: float = 0.82,
    diversity_lambda: float = 0.18,
) -> list[ScoredIdea]:
    """
    Select a small execution batch from ranked ideas.

    The procedure mirrors Algorithm 2:
    - assign ideas to niche cells,
    - remove archive-level and within-cell near-duplicates,
    - repeatedly pick the niche with best score plus marginal diversity.
    """
    archive_ids = archive.idea_ids()
    cells: dict[str, list[ScoredIdea]] = {}

    for scored in sorted(scored_ideas, key=lambda item: item.expected_score, reverse=True):
        if scored.idea.idea_id in archive_ids:
            continue
        if scored.archive_similarity >= redundancy_threshold:
            continue
        cell = cells.setdefault(scored.idea.niche, [])
        _insert_nonredundant(cell, scored, redundancy_threshold)

    for cell in cells.values():
        cell.sort(key=lambda item: item.expected_score, reverse=True)

    selected: list[ScoredIdea] = []
    while len(selected) < k:
        best_niche = ""
        best_value = float("-inf")
        for niche, cell in cells.items():
            if not cell:
                continue
            candidate = cell[0]
            value = candidate.expected_score + diversity_lambda * _marginal_diversity(candidate, selected)
            if value > best_value:
                best_niche = niche
                best_value = value

        if not best_niche:
            break

        selected.append(cells[best_niche].pop(0))
        if not cells[best_niche]:
            del cells[best_niche]

    return selected


def default_literature() -> list[LiteratureItem]:
    """Local evidence snippets relevant to this repository's train.py surface."""
    return [
        LiteratureItem(
            title="Muon optimizer for matrix parameters",
            summary="Use Muon-style orthogonalized updates for 2D matrices while AdamW handles embeddings and scalar parameters.",
            tags=("optimizer", "muon", "adamw", "matrix_lr"),
        ),
        LiteratureItem(
            title="Learning-rate warmdown under fixed wall-clock budget",
            summary="A cosine or linear warmdown can trade late training speed for a lower validation loss at the fixed five minute endpoint.",
            tags=("lr_schedule", "warmdown", "fixed_budget"),
        ),
        LiteratureItem(
            title="Sliding window attention pattern",
            summary="Alternating short and long attention windows can reduce attention cost while preserving occasional full context layers.",
            tags=("attention", "window_pattern", "throughput"),
        ),
        LiteratureItem(
            title="Depth-width tradeoff for small GPT runs",
            summary="Changing depth and aspect ratio shifts capacity and throughput, so fixed-time experiments should compare quality per second.",
            tags=("architecture", "depth", "width", "throughput"),
        ),
        LiteratureItem(
            title="Value residual embeddings",
            summary="Extra value embeddings can improve residual information flow, but add memory and optimizer state.",
            tags=("architecture", "value_embeddings", "memory"),
        ),
        LiteratureItem(
            title="Batch size and gradient accumulation",
            summary="The product of device batch size and accumulation controls optimizer noise, throughput, and memory pressure.",
            tags=("batching", "grad_accum", "memory"),
        ),
    ]


def format_score_table(scored_ideas: Sequence[ScoredIdea]) -> str:
    header = "rank idea_id niche score lit_sim arc_sim title"
    rows = [header]
    for rank, scored in enumerate(scored_ideas, start=1):
        rows.append(
            f"{rank:>4} {scored.idea.idea_id:<22} {scored.idea.niche:<14} "
            f"{scored.expected_score:>6.3f} {scored.literature_similarity:>7.3f} "
            f"{scored.archive_similarity:>7.3f} {scored.idea.title}"
        )
    return "\n".join(rows)
