# Research Planner Validation

This repo's original baseline is still the `program.md -> train.py -> results.tsv`
loop. The added planner layer sits above that loop and decides which candidate
ideas should spend GPU time.

## Architecture

1. Generate candidate ideas for `train.py` changes.
2. Score them with `HeuristicRewardModel` in `research_planner.py`.
3. Remove archive duplicates and pick a diverse execution batch with
   `select_for_execution`.
4. Execute the selected ideas.
5. Record the real reward, usually `val_bpb` delta, back into the archive.

The bootstrap reward is intentionally heuristic:

```text
score =
  usefulness + feasibility + evidence + novelty + simplicity
  - redundancy - risk
```

After enough real experiments exist, the feature vector can be replaced by a
learned model trained on actual `val_bpb` deltas.

## Local Feasibility Check

Run a tiny CPU proxy loop:

```bash
python toy_research_validation.py --fresh --k 3 --steps 80
```

This writes:

```text
planner_runs/toy/toy_results.tsv
planner_runs/toy/idea_archive.jsonl
```

The proxy training task is deliberately small. It verifies wiring and archive
behavior, not whether an idea will improve the real GPT run.

## 4080 Transfer Workflow

On the laptop GPU:

1. Run `python toy_research_validation.py --fresh` to sanity-check selection.
2. Pick one selected idea and translate its `patch_plan` into `train.py`.
3. Run the real baseline/experiment with `uv run train.py`.
4. Append `val_bpb`, memory, status, and patch notes to the archive or
   `results.tsv`.
5. Re-run the planner so duplicate ideas are deprioritized.
