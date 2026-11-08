"""Freeze generated night/day pairs without modifying the generation campaign.

Only committed result ledgers are consumed. Pending upstream content review is
preserved: the user's authorization to train on these images does not turn a
``pending_verification`` receipt into an upstream quality approval.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
SPLIT_SEED = 2026
VAL_FRACTION = .05
REVIEW_FIELDS = ("status", "content_status", "passed", "eligible_for_training",
                 "base_quality_passed", "quality_version", "verification_pending")


class NightDataIntegrityError(RuntimeError):
    """A committed receipt, image hash, cache label or frozen input conflicts."""


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def _write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _config(config):
    roots = sorted({str(Path(p).expanduser().resolve()) for p in config["night_roots"]})
    if not roots:
        raise ValueError("night_roots must contain at least one generation directory")
    return {"schema_version": SCHEMA_VERSION, "night_roots": roots,
            "base_cache": str(Path(config.get("base_cache", "artifacts/gallery_cache")).resolve()),
            "split_seed": SPLIT_SEED, "validation_fraction": VAL_FRACTION,
            "pending_review_policy": "user_authorized; preserve upstream review fields"}


def _base_identity(cache):
    cache = Path(cache)
    required = ("manifest.json", "paths.npy", "labels.npy", "place_keys.json", "provenance.json")
    identity = {name: _sha256(cache / name) for name in required}
    features = cache / "train_features.npy"
    bank = np.load(features, mmap_mode="r")
    stat = features.stat()
    identity["train_features.npy"] = {"path": str(features.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "shape": list(bank.shape), "dtype": str(bank.dtype)}
    return identity


def _immutable_metadata(roots):
    names = ("state/cohort_config.json", "state/sources.jsonl", "state/night_prompts.jsonl",
             "state/prompts_complete.json", "cluster.json")
    return {str(path): _sha256(path) for root in roots for name in names
            if (path := Path(root) / name).is_file()}


def _ledgers(roots):
    found = set()
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(f"Night generation directory is missing: {root}")
        # incoming/ is a duplicate transport staging area, never a published view.
        for pattern in ("shards/*/state/results.jsonl", "nodes/*/shards/*/state/results.jsonl",
                        "state/results.jsonl", "results.jsonl"):
            found.update(path.resolve() for path in root.glob(pattern))
    if not found:
        raise FileNotFoundError("No committed night-generation results.jsonl ledgers found")
    return sorted(found)


def _copy_metadata(stage, roots, ledgers):
    """Copy the exact bytes used for the snapshot, including append-only ledgers."""
    paths = set(ledgers)
    root_names = ("cluster.json", "ownership.json", "status.json", "state/cohort_config.json",
                 "state/sources.jsonl", "state/night_prompts.jsonl", "state/prompts_complete.json",
                 "state/cluster_ownership.json", "nodes/local/queue.json", "nodes/gpu2/queue.json")
    for root in map(Path, roots):
        paths.update(root / name for name in root_names if (root / name).is_file())
    for ledger in ledgers:
        paths.update(ledger.parent / name for name in ("plan_config.json", "summary.json")
                     if (ledger.parent / name).is_file())
    output = stage / "metadata"
    output.mkdir()
    index = []
    copied = {}
    for number, path in enumerate(sorted(paths)):
        payload = path.read_bytes()
        relative = f"metadata/{number:05d}_{path.name}"
        (stage / relative).write_bytes(payload)
        item = {"source_path": str(path), "snapshot_path": relative, "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
        index.append(item)
        copied[path] = stage / relative
    _write_json(output / "index.json", index)
    return copied, index


def _read_receipts(copied, ledgers, excluded):
    records = []
    by_sample = {}
    for original in ledgers:
        lines = copied[original].read_bytes().splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            origin = {"ledger": str(original), "ledger_line": number}
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if number != len(lines):
                    raise NightDataIntegrityError(f"Malformed committed ledger line: {original}:{number}")
                excluded.append({**origin, "reason": "incomplete_ledger_tail"})
                continue
            required = ("sample_id", "city", "place_id", "source_path", "source_sha256",
                        "source_dimensions", "output_path", "output_sha256")
            if any(not row.get(key) for key in required if key != "place_id") or "place_id" not in row:
                excluded.append({**origin, "sample_id": row.get("sample_id"),
                                 "reason": "incomplete_generation_result"})
                continue
            if "result_sha256" in row and row["result_sha256"] != _fingerprint(
                    {key: value for key, value in row.items() if key != "result_sha256"}):
                raise NightDataIntegrityError(f"Committed result seal differs: {original}:{number}")
            previous = by_sample.get(row["sample_id"])
            if previous is not None:
                if previous != row:
                    raise NightDataIntegrityError(f"Conflicting receipts for sample {row['sample_id']}")
                excluded.append({**origin, "sample_id": row["sample_id"], "reason": "duplicate_ledger_entry"})
                continue
            by_sample[row["sample_id"]] = row
            records.append((row, origin))
    return records


def _join_cache(receipts, config, excluded):
    cache = Path(config["base_cache"])
    metadata = _json(cache / "manifest.json")
    gsv_root = Path(metadata["root"]).resolve()
    paths = np.load(cache / "paths.npy", mmap_mode="r")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    places = _json(cache / "place_keys.json")
    features = np.load(cache / "train_features.npy", mmap_mode="r")
    if len(paths) != len(labels) or features.shape[0] != len(paths):
        raise NightDataIntegrityError("Base gallery-cache image, label and feature counts differ")
    wanted = {str(Path(row["source_path"]).resolve()) for row, _ in receipts}
    lookup = {}
    for index, value in enumerate(paths):
        relative = value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        # The cache stores exact released image names. Resolving every one of
        # 529k paths would add millions of filesystem lookups before the audit.
        full = str(gsv_root / relative)
        if full in wanted:
            if full in lookup:
                raise NightDataIntegrityError(f"Base cache reuses a source image: {full}")
            lookup[full] = index
    joined = []
    for row, origin in receipts:
        source = str(Path(row["source_path"]).resolve())
        if source not in lookup:
            raise NightDataIntegrityError(f"Night source is absent from the exact base-cache image list: {source}")
        index = lookup[source]
        label = int(labels[index])
        if not 0 <= label < len(places) or places[label] != [row["city"], int(row["place_id"])]:
            raise NightDataIntegrityError(f"Night city/place label conflicts with source cache row {index}")
        night = str(Path(row["output_path"]).resolve())
        missing = [key for key, path in (("source", source), ("night", night)) if not Path(path).is_file()]
        if missing:
            excluded.append({**origin, "sample_id": row["sample_id"],
                             "reason": "missing_" + "_and_".join(missing) + "_image"})
            continue
        result = {"sample_id": row["sample_id"], "city": row["city"], "place_id": int(row["place_id"]),
                  "source_index": index, "label": label, "source_path": source,
                  "source_sha256": row["source_sha256"], "night_path": night,
                  "night_sha256": row["output_sha256"], "source_dimensions": row["source_dimensions"],
                  **origin, **{key: row[key] for key in REVIEW_FIELDS if key in row}}
        for key in ("source_sample_id", "pair_id", "seed", "style_id", "raw_output_path",
                    "raw_output_sha256", "metadata_sha256", "result_sha256", "record_sha256"):
            if key in row:
                result[key] = row[key]
        joined.append(result)
    # A source may have repeated receipts, but never conflicting labels or hashes.
    identities = {"source_path": {}, "night_path": {}, "night_sha256": {}}
    for row in joined:
        for key, table in identities.items():
            if key == "source_path":
                identity = (row["label"], row["source_sha256"])
            elif key == "night_path":
                identity = (row["label"], row["source_path"], row["night_sha256"])
            else:
                identity = (row["label"], row["source_path"])
            previous = table.setdefault(row[key], identity)
            if previous != identity:
                raise NightDataIntegrityError(f"Conflicting {key} identity: {row[key]}")
    seen_source, seen_path, seen_hash = {}, {}, {}
    unique = []
    for row in sorted(joined, key=lambda r: (r["source_index"], r["sample_id"])):
        source_key = row["source_path"]
        if source_key in seen_source:
            previous = seen_source[source_key]
            if (previous["label"], previous["source_sha256"]) != (row["label"], row["source_sha256"]):
                raise NightDataIntegrityError(f"Conflicting source identity: {source_key}")
            excluded.append({"sample_id": row["sample_id"], "reason": "duplicate_source_image"})
            continue
        for key, table in (("night_path", seen_path), ("night_sha256", seen_hash)):
            if row[key] in table:
                previous = table[row[key]]
                if (previous["label"], previous["source_path"], previous["night_sha256"]) != (
                        row["label"], row["source_path"], row["night_sha256"]):
                    raise NightDataIntegrityError(f"Conflicting generated-image identity: {row[key]}")
                excluded.append({"sample_id": row["sample_id"], "reason": "duplicate_generated_image"})
                break
        else:
            seen_source[source_key] = row
            seen_path[row["night_path"]] = row
            seen_hash[row["night_sha256"]] = row
            unique.append(row)
    return unique


def _audit_pair(row):
    result = dict(row)
    for role in ("source", "night"):
        path = Path(row[f"{role}_path"])
        if not path.is_file():
            return None, {"sample_id": row["sample_id"], "reason": f"missing_{role}_image"}
        before = path.stat()
        actual = _sha256(path)
        if actual != row[f"{role}_sha256"]:
            raise NightDataIntegrityError(f"{role} image SHA256 differs for {row['sample_id']}: {path}")
        try:
            with Image.open(path) as image:
                image.load()
                dimensions = list(image.size)
                image_format, mode = image.format, image.mode
        except (OSError, ValueError) as exc:
            return None, {"sample_id": row["sample_id"], "reason": f"undecodable_{role}_image",
                          "detail": f"{type(exc).__name__}: {exc}"}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise NightDataIntegrityError(f"Image changed during audit: {path}")
        if dimensions != list(row["source_dimensions"]):
            return None, {"sample_id": row["sample_id"], "reason": f"{role}_dimensions_mismatch",
                          "expected": row["source_dimensions"], "actual": dimensions}
        if role == "night" and image_format != "PNG":
            return None, {"sample_id": row["sample_id"], "reason": "night_format_is_not_png"}
        result.update({f"{role}_size": after.st_size, f"{role}_mtime_ns": after.st_mtime_ns,
                       f"{role}_dimensions": dimensions, f"{role}_format": image_format, f"{role}_mode": mode})
    return result, None


def _split(records):
    places = {(row["city"], row["place_id"]) for row in records}
    ordered = sorted(places, key=lambda place: (_fingerprint([SPLIT_SEED, list(place)]), place))
    count = max(1, int(len(ordered) * VAL_FRACTION + .5)) if len(ordered) > 1 else 0
    validation = set(ordered[:count])
    for row in records:
        row["split"] = "val" if (row["city"], row["place_id"]) in validation else "train"
    train_sources = {row["source_index"] for row in records if row["split"] == "train"}
    val_sources = {row["source_index"] for row in records if row["split"] == "val"}
    if train_sources & val_sources:
        raise NightDataIntegrityError("A source image crosses the frozen night train/val split")
    return {"total": len(places), "train": len(places) - count, "val": count}


def _validate_frozen(config, directory, summary):
    if summary["input_config"] != _config(config):
        raise NightDataIntegrityError("Frozen night input configuration differs; choose a new input_dir")
    if summary["base_cache_identity"] != _base_identity(summary["input_config"]["base_cache"]):
        raise NightDataIntegrityError("Frozen night source cache identity differs")
    for path, expected in summary["immutable_upstream_sha256"].items():
        if not Path(path).is_file() or _sha256(path) != expected:
            raise NightDataIntegrityError(f"Frozen generation input identity differs: {path}")
    for filename, expected in summary["output_sha256"].items():
        if _sha256(directory / filename) != expected:
            raise NightDataIntegrityError(f"Frozen night snapshot file differs: {filename}")


def build_snapshot(config):
    """Audit and atomically freeze all currently published completed image pairs.

    Four CPU workers hash/decode images. This function does not load any model,
    use CUDA, copy image bytes, mutate a generation ledger or approve its review.
    Reusing a completed input_dir validates and returns its existing snapshot;
    later generated images do not enter that snapshot.
    """
    requested = _config(config)
    directory = Path(config["input_dir"]).expanduser().resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / ("." + directory.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "summary.json").is_file():
            summary = _json(directory / "summary.json")
            _validate_frozen(config, directory, summary)
            return summary
        if directory.exists() and any(directory.iterdir()):
            raise NightDataIntegrityError("Incomplete nonempty night input_dir; preserve it and choose a new directory")
        started = time.monotonic()
        base_identity = _base_identity(requested["base_cache"])
        immutable = _immutable_metadata(requested["night_roots"])
        ledgers = _ledgers(requested["night_roots"])
        print(f"night snapshot: freezing {len(ledgers)} completed-result ledgers", flush=True)
        with tempfile.TemporaryDirectory(prefix="." + directory.name + "-", dir=directory.parent) as temporary:
            stage = Path(temporary) / "snapshot"
            stage.mkdir()
            copied, metadata = _copy_metadata(stage, requested["night_roots"], ledgers)
            excluded = []
            receipts = _read_receipts(copied, ledgers, excluded)
            candidates = _join_cache(receipts, requested, excluded)
            records = []
            with ThreadPoolExecutor(max_workers=4) as workers:
                for number, (record, reason) in enumerate(workers.map(_audit_pair, candidates), 1):
                    if reason:
                        excluded.append(reason)
                    else:
                        records.append(record)
                    if number % 500 == 0 or number == len(candidates):
                        print(f"night snapshot: audited {number}/{len(candidates)}, "
                              f"included={len(records)}, excluded={len(excluded)}", flush=True)
            if not records:
                raise NightDataIntegrityError("No valid generated night/day pairs remain after audit")
            places = _split(records)
            _write_rows(stage / "records.jsonl", records)
            _write_rows(stage / "excluded.jsonl", excluded)
            _write_json(stage / "input_config.json", requested)
            output_hashes = {name: _sha256(stage / name) for name in
                             ("records.jsonl", "excluded.jsonl", "input_config.json", "metadata/index.json")}
            counts = Counter(row["split"] for row in records)
            summary = {"schema_version": SCHEMA_VERSION, "input_config": requested,
                "input_dir": str(directory), "records_path": str(directory / "records.jsonl"),
                "total": len(records), "train": counts["train"], "val": counts["val"],
                "counts": {"included": len(records), "train": counts["train"], "val": counts["val"],
                           "excluded": len(excluded), "receipt_rows": len(receipts)},
                "places": places, "excluded_reasons": dict(Counter(row["reason"] for row in excluded)),
                "review_status_counts": dict(Counter(str(row.get("status", "unspecified")) for row in records)),
                "base_cache_identity": base_identity, "immutable_upstream_sha256": immutable,
                "output_sha256": output_hashes, "metadata_files": len(metadata),
                "metadata_bytes": sum(row["size"] for row in metadata),
                "manifest_sha256": output_hashes["records.jsonl"], "created_unix": time.time(),
                "audit_seconds": time.monotonic() - started, "audit_workers": 4,
                "validation_scope": "Held out synthetic appearances by place for this finetuning only; "
                    "the pretrained checkpoint has already seen the corresponding GSV daytime places.",
                "pending_review_policy": requested["pending_review_policy"]}
            # Check immutable inputs again so a source change cannot straddle the snapshot.
            if immutable != _immutable_metadata(requested["night_roots"]):
                raise NightDataIntegrityError("Generation source identity changed during snapshot construction")
            if base_identity != _base_identity(requested["base_cache"]):
                raise NightDataIntegrityError("Base gallery cache changed during snapshot construction")
            _write_json(stage / "summary.json", summary)
            if directory.exists():
                directory.rmdir()  # Only the explicitly permitted empty directory.
            stage.replace(directory)
        return summary


def load_records(config, split):
    """Load train/val/all records from a complete, verified frozen input snapshot."""
    if split not in {"train", "val", "all"}:
        raise ValueError("split must be train, val or all")
    directory = Path(config["input_dir"]).expanduser().resolve()
    summary = _json(directory / "summary.json")
    _validate_frozen(config, directory, summary)
    records = []
    with (directory / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if split != "all" and row["split"] != split:
                continue
            for role in ("source", "night"):
                path = Path(row[f"{role}_path"])
                if not path.is_file():
                    raise NightDataIntegrityError(f"Frozen {role} image is missing: {path}")
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (row[f"{role}_size"], row[f"{role}_mtime_ns"]):
                    raise NightDataIntegrityError(f"Frozen {role} image changed since the SHA/decode audit: {path}")
            records.append(row)
    return records
