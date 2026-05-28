# Autoresearch Planner Feasibility Report

## 1. Objective

This project evaluates whether an autonomous research agent can use a structured planning layer to choose better experiments before spending GPU time. The original `autoresearch` baseline repeatedly edits `train.py`, runs a fixed-budget training job, and keeps or discards the change based on validation bits-per-byte (`val_bpb`). My extension adds a research-planning layer above that loop.

The goal of this feasibility run is not to claim a final improvement on the full GPT training benchmark. Instead, it validates the following mechanism end to end:

1. Generate candidate research ideas.
2. Score each idea before execution.
3. Store ideas in a structured archive by niche.
4. Select a diverse batch of promising ideas.
5. Execute a small training proxy locally.
6. Update the archive with observed outcomes.

This gives an executable prototype of the reward/archive design and produces initial evidence that the planning loop works.

## 2. Baseline System

The original repository is intentionally minimal:

- `prepare.py`: fixed data preparation, tokenizer, dataloader, and evaluation.
- `train.py`: single-file GPT training implementation modified by agents.
- `program.md`: instructions for autonomous experiment loops.

The true benchmark metric is `val_bpb`, produced by the fixed evaluator in `prepare.py`. Lower is better. Each real training experiment is intended to run for a fixed 5-minute wall-clock budget on a CUDA GPU.

## 3. Proposed Design

The added planner introduces two components:

1. A reward-style idea scorer that ranks candidate experiments before execution.
2. An archive selector that keeps track of previously attempted ideas and promotes diversity across experiment categories.

The main design motivation is that a naive autonomous loop can repeatedly try similar ideas or spend GPU time on experiments that are novel but unlikely to be feasible. The planner makes the selection process explicit and inspectable.

## 4. Algorithm 1: Idea Reward Model

The first stage scores each candidate idea using both prior evidence and archive redundancy.

### Pseudocode

```text
Input:
  research topic T
  candidate ideas I
  literature/evidence database L
  idea archive A
  reward model R_phi

Output:
  ranked candidate ideas I_ranked

for each idea i in I:
    P_i <- retrieve_related_evidence(L, i)

    literature_similarity <- max_similarity(i, P_i)
    archive_similarity <- max_similarity(i, A)

    novelty <- 1 - literature_similarity
    evidence <- literature_similarity
    redundancy <- archive_similarity

    usefulness, feasibility, risk, effort <- judge(i, T)
    simplicity <- 1 - effort

    score_i <- R_phi(
        usefulness,
        feasibility,
        evidence,
        novelty,
        simplicity,
        redundancy,
        risk
    )

return sort(I, by=score_i, descending=True)
```

### Practical Scoring Function

The prototype uses a bootstrap heuristic:

```text
score =
    usefulness
  + feasibility
  + evidence
  + novelty
  + simplicity
  - redundancy
  - risk
```

This is implemented in `research_planner.py` as `HeuristicRewardModel`. A learned reward model can later replace this heuristic once enough real experiments have been collected.

## 5. Algorithm 2: Archive-Based Selection

The second stage uses the ranked ideas to select a non-redundant and diverse execution set.

### Pseudocode

```text
Input:
  ranked ideas I_ranked
  archive A
  niche set N
  redundancy threshold t
  execution limit K
  diversity weight lambda

Output:
  selected execution ideas I_exec

for each idea i in I_ranked:
    n_i <- classify(i, N)
    A[n_i] <- A[n_i] union {i}

    remove near-duplicates in A[n_i]:
        if similarity(i, j) >= t:
            keep the idea with higher predicted score

I_exec <- empty set

while |I_exec| < K:
    for each niche n in N:
        best_score_n <- max predicted score among unselected ideas in A[n]
        diversity_bonus_n <- marginal_diversity(A[n], I_exec)
        niche_value_n <- best_score_n + lambda * diversity_bonus_n

    n_star <- argmax niche_value_n
    i_star <- best unselected idea in A[n_star]
    I_exec <- I_exec union {i_star}
    mark i_star as selected

return I_exec
```

This is implemented as `select_for_execution` in `research_planner.py`.

## 6. Implementation Summary

The prototype adds the following files:

- `research_planner.py`: core idea, scoring, similarity, archive, and selection logic.
- `toy_research_validation.py`: small CPU proxy training loop for end-to-end validation.
- `PLANNER.md`: short usage and transfer notes.
- `deliverable_runs/final_20260528/`: final run outputs.

The candidate idea representation includes:

- `idea_id`
- `title`
- `description`
- `niche`
- `patch_plan`
- `tags`
- `prior`
- `risk`
- `effort`

The archive stores:

- predicted score
- niche
- status
- metric name
- metric value
- metric delta
- notes

## 7. Candidate Ideas Tested

The local validation used six candidate ideas:

| idea_id | niche | high-level idea |
| --- | --- | --- |
| lower_lr_cosine | lr_schedule | lower learning rate with cosine warmdown |
| adamw_weight_decay | optimizer | add moderate AdamW weight decay |
| smaller_faster_model | efficiency | trade capacity for more steps |
| wider_hidden | architecture | increase hidden width |
| dropout_regularization | regularization | add light dropout |
| sgd_momentum | optimizer | explore SGD momentum |

These ideas are proxies for the kinds of changes that could later be translated into the full `train.py` GPT training script.

## 8. Local Validation Setup

The local machine is a Mac environment without CUDA or MPS support in the current PyTorch installation:

```text
PyTorch: 2.11.0
CUDA available: false
MPS available: false
```

Because the original `train.py` depends on CUDA-oriented kernels, full GPT training was not run locally. Instead, the project runs a complete CPU proxy validation through `toy_research_validation.py`.

The proxy task is intentionally small but still exercises the full planner loop:

```text
candidate ideas -> scoring -> niche/archive selection -> training -> keep/discard -> archive update
```

Command used:

```bash
python toy_research_validation.py --out-dir deliverable_runs/final_20260528 --fresh --k 6 --steps 5000
```

Wall-clock runtime:

```text
10.63 seconds
```

## 9. Initial Results

The baseline proxy validation loss was:

```text
baseline toy_val_loss = 1.588353
```

Full result table:

| idea_id | niche | predicted_score | toy_val_loss | delta_vs_baseline | status |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | baseline | | 1.588353 | 0.000000 | baseline |
| lower_lr_cosine | lr_schedule | 0.589915 | 0.848085 | +0.740269 | keep |
| adamw_weight_decay | optimizer | 0.516268 | 1.251814 | +0.336540 | keep |
| smaller_faster_model | efficiency | 0.477584 | 1.228357 | +0.359996 | keep |
| wider_hidden | architecture | 0.472088 | 1.804444 | -0.216090 | discard |
| dropout_regularization | regularization | 0.368623 | 1.022214 | +0.566140 | keep |
| sgd_momentum | optimizer | 0.290354 | 2.729451 | -1.141098 | discard |

Four of the six selected ideas improved the proxy metric relative to baseline, while two were correctly recorded as failed directions for the archive.

## 10. Interpretation

The result demonstrates that the proposed planner is operational:

1. It can rank heterogeneous candidate ideas.
2. It can select ideas across multiple niches.
3. It can execute a training proxy without manual intervention.
4. It can record structured outcomes in an archive.
5. It can distinguish useful ideas from unsuccessful ideas in the local proxy setting.

The experiment does not yet prove improvement on the full GPT training benchmark. It does, however, validate the software architecture needed to run such experiments more systematically once GPU resources are available.

## 11. GPU Cluster Status

The next step would normally be to run selected ideas on a CUDA GPU, such as the University of Michigan Great Lakes cluster or a local NVIDIA GPU laptop. That would allow the selected ideas to be translated into `train.py` patches and evaluated using the real `val_bpb` metric.

At this stage, connecting to the school GPU cluster is temporarily inconvenient because it requires VPN access, environment setup, file transfer, and queue submission. To avoid blocking the feasibility validation on cluster logistics, I completed the full local planner/proxy training loop instead. This gives a complete and reproducible local run while leaving full GPU training as the next experimental stage.

## 12. Next Steps

The recommended next stage is:

1. Transfer the repository to a CUDA GPU environment.
2. Run `uv sync` and `uv run prepare.py --num-shards 2` for a small data setup.
3. Start from the selected ideas marked `keep`.
4. Translate each idea's `patch_plan` into a controlled `train.py` change.
5. Run `uv run train.py` and record `val_bpb`.
6. Feed real `val_bpb` deltas back into the archive.

Once enough real experiments are collected, the heuristic reward model can be replaced with a learned model trained on actual experiment outcomes.

## 13. Reproducibility Files

Final local deliverables:

- `deliverable_runs/final_20260528/SUMMARY.md`
- `deliverable_runs/final_20260528/toy_results.tsv`
- `deliverable_runs/final_20260528/idea_archive.jsonl`
- `research_planner.py`
- `toy_research_validation.py`
- `PLANNER.md`

The run can be reproduced locally with:

```bash
python toy_research_validation.py --out-dir deliverable_runs/final_20260528 --fresh --k 6 --steps 5000
```
