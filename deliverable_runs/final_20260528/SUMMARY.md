# Autoresearch Planner Feasibility Run

Date: 2026-05-28

## Goal

Validate a research-planning layer for autonomous experiment selection:

1. Generate candidate training ideas.
2. Score ideas by usefulness, feasibility, evidence, novelty, simplicity, redundancy, and risk.
3. Place ideas into an archive organized by niche.
4. Select a diverse execution batch.
5. Run a small proxy training task and update the archive with keep/discard outcomes.

This run validates the planning and archive mechanics locally. It does not replace the full CUDA GPT training loop in `train.py`.

## Environment

- Machine: local Mac, arm64
- PyTorch: 2.11.0
- CUDA available: false
- MPS available: false
- Training path used: CPU toy proxy validation

## Command

```bash
python toy_research_validation.py --out-dir deliverable_runs/final_20260528 --fresh --k 6 --steps 5000
```

Runtime: 10.63 seconds wall-clock.

## Results

| idea_id | niche | predicted_score | toy_val_loss | delta_vs_baseline | status |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | baseline | | 1.588353 | 0.000000 | baseline |
| lower_lr_cosine | lr_schedule | 0.589915 | 0.848085 | +0.740269 | keep |
| adamw_weight_decay | optimizer | 0.516268 | 1.251814 | +0.336540 | keep |
| smaller_faster_model | efficiency | 0.477584 | 1.228357 | +0.359996 | keep |
| wider_hidden | architecture | 0.472088 | 1.804444 | -0.216090 | discard |
| dropout_regularization | regularization | 0.368623 | 1.022214 | +0.566140 | keep |
| sgd_momentum | optimizer | 0.290354 | 2.729451 | -1.141098 | discard |

## Takeaway

The planner successfully ranked candidate ideas, selected all six non-duplicate candidates across niches, executed the proxy training task, and wrote structured outcomes to the archive.

Generated files:

- `toy_results.tsv`: tabular experiment outcomes.
- `idea_archive.jsonl`: structured archive entries with predicted scores and observed deltas.
