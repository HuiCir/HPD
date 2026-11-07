# TRAJECT-Bench Split Reproduction

This directory publishes the complete case-level split index and the inputs
needed to reproduce it. The benchmark data itself is not duplicated here.

## Published Split

The pinned public snapshot contains 5,870 accepted task instances. The split
is assigned to 3,707 groups after co-grouping parallel simple/hard pairs and
sequential cases with identical normalized execution traces.

| Split | Groups | Cases |
| --- | ---: | ---: |
| Train | 2,965 | 4,703 |
| Validation | 371 | 584 |
| Test | 371 | 583 |

`split_index.csv` has one row per case and records its `case_id`,
`group_id`, split, source file and source row. `split_config.json` defines
the source revision, filters, seed, grouping, stratification, quota rounding
and hash ordering. `source_manifest.json` pins every input file by byte size
and SHA-256.

## Reproduce

Download the public source at the pinned revision, then run the self-contained
standard-library script:

```bash
git clone https://github.com/PengfeiHePower/TRAJECT-Bench.git
git -C TRAJECT-Bench checkout 2723fd890778dbfb6af9e3aa8ee1c22272979468
python3 reproducibility/trajectory_bench_split/prepare_trajectory_bench.py \
  --source TRAJECT-Bench/public_data \
  --config reproducibility/trajectory_bench_split/split_config.json \
  --output ./trajectory_bench_rebuilt
```

Run this command from the HPD repository root. The script verifies the
downloaded files against the pinned manifest before processing. It writes
`processed/train.jsonl.gz`, `validation.jsonl.gz`, `test.jsonl.gz`,
`groups.jsonl`, and `split_index.csv` under the output directory. The
generated case-level index should match the published CSV byte-for-byte for
the pinned source and config.

## Assignment Rules

Rows with an empty query, final answer, or tool trajectory, a missing tool
name, or an explicit non-executable marker are excluded. Warnings such as
failed or empty tool results are retained; they do not remove a case.

Parallel simple/hard rows are kept in one group. Sequential rows in the same
domain with identical normalized tool names, parameters, execution statuses,
and outputs are co-grouped. Groups are stratified by trajectory type, domain,
and maximum group tool count (exact buckets through 10, then `11_plus`).

With seed `20260714`, the script computes per-stratum held-out quotas using
the largest-remainder rule. Stable SHA-256 ordering assigns the first quota
to test, the next quota to validation, and all remaining groups to train.
The complete deterministic details are encoded in `split_config.json`.
