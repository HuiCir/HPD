#!/usr/bin/env python3
"""Vendor, audit, normalize, and split the complete public TRAJECT-Bench data."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT = Path.cwd() / "trajectory_bench_rebuilt"
DEFAULT_SPLIT_CONFIG = Path(__file__).resolve().with_name("split_config.json")
PARALLEL_DIFFICULTIES = ("simple_ver", "hard_ver")
SPLITS = ("train", "validation", "test")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format") != "pi-traject-bench-split-config-v1":
        raise ValueError(f"unsupported split config format: {config.get('format')}")
    split = config.get("split", {})
    ratios = split.get("ratios", {})
    if set(ratios) != set(SPLITS):
        raise ValueError(f"split.ratios must define exactly {SPLITS}")
    if abs(sum(float(ratios[name]) for name in SPLITS) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    if any(not 0.0 < float(ratios[name]) < 1.0 for name in SPLITS):
        raise ValueError("each split ratio must be between 0 and 1")
    if float(ratios["validation"]) != float(ratios["test"]):
        raise ValueError("this split algorithm requires equal validation and test ratios")
    if float(ratios["train"]) + float(ratios["validation"]) + float(ratios["test"]) != 1.0:
        raise ValueError("invalid split ratios")
    algorithm = split.get("algorithm")
    if algorithm != "sha256_stratified_group_v1":
        raise ValueError(f"unsupported split algorithm: {algorithm}")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("config seed must be non-negative")
    if int(config.get("processing", {}).get("active_tool_pool_cap", 0)) <= 0:
        raise ValueError("processing.active_tool_pool_cap must be positive")
    return config


def compact_parameter(parameter: Any) -> dict[str, Any]:
    if not isinstance(parameter, dict):
        return {}
    return {
        key: parameter[key]
        for key in ("name", "value", "type", "default", "description")
        if key in parameter and parameter[key] not in (None, "")
    }


def normalize_tool(tool: Any, trajectory_type: str) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {"name": "", "description": "", "required_parameters": [], "optional_parameters": []}
    required = tool.get("required parameters", tool.get("required_parameters", [])) or []
    optional = tool.get("optional parameters", tool.get("optional_parameters", [])) or []
    status = str(tool.get("execution_status") or "success").lower()
    return {
        "name": str(tool.get("tool name") or tool.get("name") or "").strip(),
        "description": str(tool.get("tool description") or tool.get("description") or "").strip(),
        "required_parameters": [item for item in (compact_parameter(value) for value in required) if item],
        "optional_parameters": [item for item in (compact_parameter(value) for value in optional) if item],
        "execution_status": status,
        "executed_output": str(tool.get("executed_output") or ""),
        "execution_mode": "parallel" if trajectory_type == "parallel" else "sequential",
        "sequence_step": tool.get("sequence_step"),
        "api_name": tool.get("API name"),
        "domain_name": tool.get("domain name"),
        "parent_tool_name": tool.get("parent tool name"),
        "adapt_parameter": tool.get("adapt_parameter"),
        "adapt_constraint": tool.get("adapt_constraint"),
        "adapt_reason": tool.get("adapt_reason"),
    }


def normalized_final(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("answer") or value.get("final_answer") or ""
    return str(value or "").strip()


def tool_signature(tools: list[dict[str, Any]]) -> str:
    payload = [
        {
            "name": tool["name"],
            "required_parameters": tool["required_parameters"],
            "optional_parameters": tool["optional_parameters"],
            "execution_status": tool["execution_status"],
            "executed_output": tool["executed_output"],
        }
        for tool in tools
    ]
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize_pool_tool(tool: Any) -> dict[str, Any]:
    normalized = normalize_tool(tool, "parallel")
    return {
        "name": normalized["name"],
        "description": normalized["description"],
        "required_parameters": normalized["required_parameters"],
        "optional_parameters": normalized["optional_parameters"],
    }


def active_tool_pool(
    *,
    case_id: str,
    gold_tools: list[dict[str, Any]],
    domain_pool: list[dict[str, Any]],
    pool_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    gold_names = {tool["name"] for tool in gold_tools}
    distractors = [tool for tool in domain_pool if tool["name"] and tool["name"] not in gold_names]
    distractors.sort(key=lambda tool: stable_hash(f"{seed}:{case_id}:{tool['name']}"))
    selected = [
        {
            "name": tool["name"],
            "description": tool["description"],
            "required_parameters": tool["required_parameters"],
            "optional_parameters": tool["optional_parameters"],
        }
        for tool in gold_tools
    ]
    selected.extend(distractors[: max(0, pool_cap - len(selected))])
    selected.sort(key=lambda tool: stable_hash(f"active:{seed}:{case_id}:{tool['name']}"))
    return selected


def audit_case(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    exclusions: list[str] = []
    warnings: list[str] = []
    tools = record["tools"]
    if not record["query"]:
        exclusions.append("empty_query")
    if not record["final_answer"]:
        exclusions.append("empty_final_answer")
    if not tools:
        exclusions.append("empty_tool_trajectory")
    if any(not tool["name"] for tool in tools):
        exclusions.append("missing_tool_name")
    if record.get("executable") is False:
        exclusions.append("marked_non_executable")
    names = [tool["name"] for tool in tools]
    if len(names) != len(set(names)):
        warnings.append("repeated_tool_name_in_trajectory")
    if any(tool["execution_status"] not in {"success", "ok", "true", "none"} for tool in tools):
        warnings.append("contains_failed_tool_result")
    if any(not tool["executed_output"] for tool in tools):
        warnings.append("contains_empty_tool_output")
    if any(
        tool.get("adapt_parameter")
        and not any("value" in parameter for parameter in tool["required_parameters"] + tool["optional_parameters"])
        for tool in tools
    ):
        warnings.append("contains_unbound_runtime_parameter")
    if not 3 <= len(tools) <= 10:
        warnings.append("tool_count_outside_published_3_to_10_range")
    return sorted(set(exclusions)), sorted(set(warnings))


def iter_source_cases(raw: Path) -> Iterable[dict[str, Any]]:
    parallel_root = raw / "parallel"
    for domain_dir in sorted(path for path in parallel_root.iterdir() if path.is_dir()):
        domain = domain_dir.name
        paired = {difficulty: read_json(domain_dir / f"{difficulty}.json") for difficulty in PARALLEL_DIFFICULTIES}
        if len(paired["simple_ver"]) != len(paired["hard_ver"]):
            raise ValueError(f"parallel pair length mismatch in {domain}")
        for index in range(len(paired["simple_ver"])):
            group_id = f"parallel:{domain}:{index:04d}"
            signatures: dict[str, str] = {}
            for difficulty in PARALLEL_DIFFICULTIES:
                source = paired[difficulty][index]
                tools = [normalize_tool(tool, "parallel") for tool in source.get("tool list", [])]
                signatures[difficulty] = tool_signature(tools)
                yield {
                    "id": f"{group_id}:{difficulty}",
                    "group_id": group_id,
                    "source_group_id": group_id,
                    "trajectory_type": "parallel",
                    "domain": domain,
                    "difficulty": difficulty.removesuffix("_ver"),
                    "source_file": f"parallel/{domain}/{difficulty}.json",
                    "source_index": index,
                    "query": str(source.get("query") or "").strip(),
                    "final_answer": normalized_final(source.get("final_answer")),
                    "final_answer_reason": (
                        str(source.get("final_answer", {}).get("reason") or "")
                        if isinstance(source.get("final_answer"), dict)
                        else ""
                    ),
                    "executable": source.get("executable", True),
                    "tools": tools,
                    "tool_signature": signatures[difficulty],
                    "pair_tool_signature": None,
                    "source_metadata": {
                        "num_tools_used": source.get("num_tools_used"),
                        "num_successful_tools": source.get("num_successful_tools"),
                        "generation_info": source.get("generation_info"),
                    },
                }

    sequential_root = raw / "sequential"
    for domain_dir in sorted(path for path in sequential_root.iterdir() if path.is_dir()):
        sources: list[tuple[str, str]] = []
        canonical = domain_dir / "traj_query.json"
        if canonical.exists():
            sources.append(("traj_query.json", "canonical"))
        legacy = domain_dir / "simple_ver.json"
        if legacy.exists():
            sources.append(("simple_ver.json", "legacy_simple"))
        for filename, difficulty in sources:
            for index, source in enumerate(read_json(domain_dir / filename)):
                raw_tools = source.get("tool list", source.get("tool_list", []))
                tools = [normalize_tool(tool, "sequential") for tool in raw_tools]
                signature = tool_signature(tools)
                source_group_id = f"sequential:{domain_dir.name}:{difficulty}:{index:04d}"
                group_id = f"sequential:{domain_dir.name}:trajectory:{signature[:20]}"
                yield {
                    "id": source_group_id,
                    "group_id": group_id,
                    "source_group_id": source_group_id,
                    "trajectory_type": "sequential",
                    "domain": domain_dir.name,
                    "difficulty": difficulty,
                    "source_file": f"sequential/{domain_dir.name}/{filename}",
                    "source_index": index,
                    "query": str(source.get("query") or "").strip(),
                    "final_answer": normalized_final(source.get("final_answer")),
                    "final_answer_reason": (
                        str(source.get("final_answer", {}).get("reason") or "")
                        if isinstance(source.get("final_answer"), dict)
                        else ""
                    ),
                    "executable": source.get("executable", True),
                    "tools": tools,
                    "tool_signature": signature,
                    "pair_tool_signature": None,
                    "source_metadata": {
                        "sequence_name": source.get("sequence_name"),
                        "sequence_description": source.get("sequence_description"),
                        "num_tools_used": source.get("num_tools_used"),
                        "num_successful_tools": source.get("num_successful_tools"),
                        "generation_info": source.get("generation_info"),
                    },
                }


def split_groups(
    records: list[dict[str, Any]],
    *,
    seed: int,
    split_ratios: dict[str, float],
) -> dict[str, str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        tool_count = max(len(member["tools"]) for member in members)
        bucket = str(tool_count) if tool_count <= 10 else "11_plus"
        first = members[0]
        strata[(first["trajectory_type"], first["domain"], bucket)].append(group_id)

    eval_ratio = split_ratios["validation"]
    target_eval_groups = int(len(groups) * eval_ratio + 0.5)
    eval_quota = {
        stratum: int(len(group_ids) * eval_ratio)
        for stratum, group_ids in strata.items()
    }
    remaining = target_eval_groups - sum(eval_quota.values())
    quota_candidates = sorted(
        strata,
        key=lambda stratum: (
            -(len(strata[stratum]) * eval_ratio % 1),
            stable_hash(f"quota:{seed}:{stratum}"),
        ),
    )
    for stratum in quota_candidates:
        if remaining <= 0:
            break
        proposed = eval_quota[stratum] + 1
        if proposed * 2 < len(strata[stratum]):
            eval_quota[stratum] = proposed
            remaining -= 1
    if remaining:
        raise RuntimeError(f"could not allocate {remaining} held-out groups across strata")

    assignments: dict[str, str] = {}
    for stratum, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda value: stable_hash(f"{seed}:{value}"))
        test_count = eval_quota[stratum]
        validation_count = eval_quota[stratum]
        for index, group_id in enumerate(ordered):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            assignments[group_id] = split
    return assignments


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_split_index(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write a portable case-to-group-to-split mapping."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "group_id",
        "split",
        "trajectory_type",
        "domain",
        "difficulty",
        "source_file",
        "source_index",
        "tool_count",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["id"]):
            writer.writerow(
                {
                    "case_id": record["id"],
                    "group_id": record["group_id"],
                    "split": record["split"],
                    "trajectory_type": record["trajectory_type"],
                    "domain": record["domain"],
                    "difficulty": record["difficulty"],
                    "source_file": record["source_file"],
                    "source_index": record["source_index"],
                    "tool_count": len(record["tools"]),
                }
            )


def vendor_raw(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace existing raw data at {destination}; use --no-vendor-raw"
        )
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".DS_Store"))


def source_manifest(raw: Path, upstream_commit: str | None) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in raw.rglob("*") if item.is_file()):
        relative = path.relative_to(raw).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {
        "format": "traject-bench-public-snapshot-v1",
        "upstream": "https://github.com/PengfeiHePower/TRAJECT-Bench",
        "upstream_commit": upstream_commit,
        "files": files,
    }


def validate_source_manifest(source: Path, manifest_path: Path) -> None:
    """Require the downloaded public snapshot to match the pinned file index."""

    expected_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = expected_manifest.get("files", {})
    actual_files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    if actual_files != set(expected_files):
        raise ValueError(
            "source snapshot file list differs from the pinned manifest: "
            f"missing={sorted(set(expected_files) - actual_files)[:5]}, "
            f"extra={sorted(actual_files - set(expected_files))[:5]}"
        )
    for relative, expected in expected_files.items():
        path = source / relative
        if path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"source snapshot byte size differs: {relative}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"source snapshot checksum differs: {relative}")


def nested_counts(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[key]) for record in records).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="upstream TRAJECT-Bench/public_data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--pool-cap", type=int, help="override the config's active tool pool cap")
    parser.add_argument("--upstream-commit")
    parser.add_argument("--no-vendor-raw", action="store_true")
    args = parser.parse_args()

    config = load_split_config(args.config.resolve())
    seed = int(config["seed"])
    split_ratios = {name: float(config["split"]["ratios"][name]) for name in SPLITS}
    pool_cap = int(
        args.pool_cap
        if args.pool_cap is not None
        else config["processing"]["active_tool_pool_cap"]
    )
    upstream_commit = args.upstream_commit or config["dataset"].get("upstream_commit")
    source = args.source.resolve()
    manifest_name = config["dataset"].get("source_manifest_file")
    if manifest_name:
        validate_source_manifest(source, args.config.resolve().parent / manifest_name)
    output = args.output.resolve()
    raw = output / "raw"
    if not args.no_vendor_raw:
        vendor_raw(source, raw)
    elif not raw.exists():
        raise FileNotFoundError(raw)

    pool_by_domain = {
        path.stem.removesuffix("_tool"): [normalize_pool_tool(tool) for tool in read_json(path)]
        for path in sorted((raw / "tools").glob("*_tool.json"))
        if path.name != "all_tools.json"
    }
    records = list(iter_source_cases(raw))
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)
    for group_records in by_group.values():
        if group_records[0]["trajectory_type"] == "parallel":
            signatures = {record["tool_signature"] for record in group_records}
            for record in group_records:
                record["pair_tool_signature"] = stable_hash(":".join(sorted(signatures)))
                if len(signatures) > 1:
                    record.setdefault("pre_audit_warnings", []).append("simple_hard_tool_trajectory_mismatch")
        elif len(group_records) > 1:
            for record in group_records:
                record.setdefault("pre_audit_warnings", []).append("duplicate_executed_trajectory_group")

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for record in records:
        exclusions, warnings = audit_case(record)
        warnings = sorted(set(warnings + record.pop("pre_audit_warnings", [])))
        audit_rows.append(
            {
                "id": record["id"],
                "group_id": record["group_id"],
                "source_file": record["source_file"],
                "source_index": record["source_index"],
                "exclusions": exclusions,
                "warnings": warnings,
            }
        )
        if exclusions:
            excluded.append({**record, "exclusion_reasons": exclusions, "warnings": warnings})
            continue
        record["warnings"] = warnings
        record["active_tools"] = active_tool_pool(
            case_id=record["id"],
            gold_tools=record["tools"],
            domain_pool=pool_by_domain.get(record["domain"], []),
            pool_cap=pool_cap,
            seed=seed,
        )
        accepted.append(record)

    assignments = split_groups(accepted, seed=seed, split_ratios=split_ratios)
    for record in accepted:
        record["split"] = assignments[record["group_id"]]

    expected = config.get("expected_snapshot", {})
    observed = {
        "source_task_instances": len(records),
        "accepted_task_instances": len(accepted),
        "excluded_task_instances": len(excluded),
        "source_groups_before_signature_coalescing": len(
            {record["source_group_id"] for record in accepted}
        ),
        "split_groups": len(assignments),
    }
    for name, value in observed.items():
        if name in expected and int(expected[name]) != value:
            raise ValueError(
                f"source snapshot differs from config at {name}: "
                f"expected {expected[name]}, observed {value}"
            )
    for split_name in SPLITS:
        split_records = [record for record in accepted if record["split"] == split_name]
        split_groups_count = len({record["group_id"] for record in split_records})
        expected_split = expected.get("splits", {}).get(split_name, {})
        if "instances" in expected_split and int(expected_split["instances"]) != len(split_records):
            raise ValueError(
                f"source snapshot differs from config at {split_name} instances: "
                f"expected {expected_split['instances']}, observed {len(split_records)}"
            )
        if "groups" in expected_split and int(expected_split["groups"]) != split_groups_count:
            raise ValueError(
                f"source snapshot differs from config at {split_name} groups: "
                f"expected {expected_split['groups']}, observed {split_groups_count}"
            )

    processed = output / "processed"
    for split in SPLITS:
        split_rows = sorted(
            (record for record in accepted if record["split"] == split),
            key=lambda record: record["id"],
        )
        write_jsonl_gz(processed / f"{split}.jsonl.gz", split_rows)
    write_jsonl_gz(processed / "all.jsonl.gz", sorted(accepted, key=lambda record: record["id"]))
    write_jsonl_gz(processed / "audit.jsonl.gz", sorted(audit_rows, key=lambda record: record["id"]))
    write_jsonl(processed / "excluded.jsonl", sorted(excluded, key=lambda record: record["id"]))

    group_rows = []
    for group_id, members in sorted(by_group.items()):
        accepted_members = [member for member in accepted if member["group_id"] == group_id]
        if not accepted_members:
            continue
        group_rows.append(
            {
                "group_id": group_id,
                "split": assignments[group_id],
                "trajectory_type": accepted_members[0]["trajectory_type"],
                "domain": accepted_members[0]["domain"],
                "case_ids": sorted(member["id"] for member in accepted_members),
                "tool_count": max(len(member["tools"]) for member in accepted_members),
            }
        )
    write_jsonl(processed / "groups.jsonl", group_rows)
    write_split_index(processed / "split_index.csv", accepted)

    warning_counts = Counter(warning for row in audit_rows for warning in row["warnings"])
    exclusion_counts = Counter(reason for row in audit_rows for reason in row["exclusions"])
    split_summary = {}
    for split in SPLITS:
        subset = [record for record in accepted if record["split"] == split]
        split_summary[split] = {
            "instances": len(subset),
            "groups": len({record["group_id"] for record in subset}),
            "by_trajectory_type": nested_counts(subset, "trajectory_type"),
            "by_domain": nested_counts(subset, "domain"),
            "by_difficulty": nested_counts(subset, "difficulty"),
            "tool_steps": sum(len(record["tools"]) for record in subset),
            "lifecycle_boundaries": sum(
                2 if record["trajectory_type"] == "parallel" else len(record["tools"]) + 1
                for record in subset
            ),
        }
    summary = {
        "format": "pi-traject-bench-splits-v1",
        "config": "split_config.json",
        "seed": seed,
        "split_ratios": split_ratios,
        "stratification": config["split"]["stratification"],
        "grouping": (
            "parallel simple/hard pairs share one group; sequential source cases with identical normalized "
            "tools, parameters, statuses, and outputs share one leakage group"
        ),
        "raw_task_instances": len(records),
        "accepted_task_instances": len(accepted),
        "excluded_task_instances": len(excluded),
        "accepted_groups": len(assignments),
        "source_task_groups": len({record["source_group_id"] for record in accepted}),
        "warning_counts": dict(sorted(warning_counts.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "splits": split_summary,
        "non_task_assets": {
            "sequential/Travel/sequence.json": "40 chain templates without user queries; retained in raw only",
            "selected_category.json": "category metadata; retained in raw only",
            "tools/*.json": "tool catalogs used to construct distractor pools; retained in raw",
            "../trajectory-bench/seeds.json": "separate IWG seed collection, not part of the public TRAJECT-Bench release",
        },
    }
    (processed / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest(raw, upstream_commit), ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
