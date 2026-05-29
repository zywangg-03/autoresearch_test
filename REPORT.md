# 前期验证报告

在 andrej karpathy autoresearch 训练循环之上增加一个研究规划层，对候选实验想法进行评分、去重、分组和选择。

验证链路：

```text
候选想法 -> 预评分 -> archive/niche 选择 -> 小规模训练验证 -> keep/discard -> 更新 archive
```

## 2. Pseudocode

### Algorithm 1: Reward Model / Idea Scoring

```text
Input:
  research topic T
  candidate ideas I
  literature/evidence database L
  idea archive A

Output:
  ranked ideas I_ranked

for each idea i in I:
    related_papers <- retrieve(L, i)

    S_lit <- max_similarity(i, related_papers)
    S_arc <- max_similarity(i, A)

    novelty <- 1 - S_lit
    evidence <- S_lit
    redundancy <- S_arc

    usefulness, feasibility, risk, effort <- judge(i, T)
    simplicity <- 1 - effort

    score_i <- reward(
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

当前实现使用如下 reward：

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

### Algorithm 2: Advanced Archive / Diverse Selection

```text
Input:
  ranked ideas I_ranked
  archive A
  niche set N
  redundancy threshold t
  execution limit K
  diversity weight lambda

Output:
  selected ideas I_exec

for each idea i in I_ranked:
    n_i <- classify(i, N)
    add i into archive cell A[n_i]

    if i is too similar to an existing idea:
        keep the one with higher predicted score

I_exec <- empty set

while |I_exec| < K:
    for each niche n:
        best_score <- max score among unselected ideas in A[n]
        diversity_bonus <- marginal_diversity(A[n], I_exec)
        niche_value <- best_score + lambda * diversity_bonus

    n_star <- argmax niche_value
    i_star <- best unselected idea in A[n_star]
    add i_star to I_exec

return I_exec
```

## 3. Implementation

- `research_planner.py`：实现 idea 数据结构、文本相似度、heuristic reward model、archive 记录和多样性选择。
- `toy_research_validation.py`：实现一个小型 CPU proxy training，用来完整测试 planner 的闭环。

最终输出文件位于：

- `final_20260528/SUMMARY.md`
- `final_20260528/toy_results.tsv`
- `final_20260528/idea_archive.jsonl`

本机运行命令：

```bash
python toy_research_validation.py --out-dir deliverable_runs/final_20260528 --fresh --k 6 --steps 5000
```

## 4. Initial Results

本次共测试 6 个候选 idea，baseline proxy validation loss 为：

```text
baseline toy_val_loss = 1.588353
```

结果如下：

| idea_id | niche | predicted_score | toy_val_loss | delta_vs_baseline | status |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | baseline | | 1.588353 | 0.000000 | baseline |
| lower_lr_cosine | lr_schedule | 0.589915 | 0.848085 | +0.740269 | keep |
| adamw_weight_decay | optimizer | 0.516268 | 1.251814 | +0.336540 | keep |
| smaller_faster_model | efficiency | 0.477584 | 1.228357 | +0.359996 | keep |
| wider_hidden | architecture | 0.472088 | 1.804444 | -0.216090 | discard |
| dropout_regularization | regularization | 0.368623 | 1.022214 | +0.566140 | keep |
| sgd_momentum | optimizer | 0.290354 | 2.729451 | -1.141098 | discard |

其中 4 个 idea 在 proxy task 上优于 baseline，被标记为 `keep`；2 个 idea 表现更差，被标记为 `discard`。这说明 archive 能记录成功和失败方向，后续选择时可以避免重复尝试相似低效方案。

## 5. 结论与后续工作

本次实验完成了前期设计验证：planner 能够对候选 idea 进行评分、按 niche 组织 archive、选择多样化实验，并根据小规模训练结果更新 keep/discard 状态。

下一步需要将该 repo 同步到学校 GPU cluster，在 CUDA 环境中把被标记为 `keep` 的 idea 转换为真实 `train.py` 修改，并用完整训练得到真实 `val_bpb` 结果。
