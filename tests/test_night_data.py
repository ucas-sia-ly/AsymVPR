import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from reproduction.night_data import (NightDataIntegrityError, _fingerprint,
                                      build_snapshot, load_records)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, count=20):
    gsv, cache, night = tmp_path / "gsv", tmp_path / "cache", tmp_path / "night"
    cache.mkdir(); night.mkdir()
    paths, rows = [], []
    for index in range(count):
        rel = f"Images/City/exact_float_name_{index}.jpg"
        source = gsv / rel
        source_sha = image(source, (index, 40, 60))
        output = night / "images" / f"night_{index}.png"
        output_sha = image(output, (index, 3, 9))
        row = dict(sample_id=f"sample{index}", city="City", place_id=index,
                   source_path=str(source), source_sha256=source_sha,
                   source_dimensions=[16, 12], output_path=str(output),
                   output_sha256=output_sha, status="pending_verification", passed=False,
                   eligible_for_training=False, verification_pending=True)
        row["result_sha256"] = _fingerprint(row)
        rows.append(row); paths.append(rel.encode())
    write_json(cache / "manifest.json", {"root": str(gsv), "signature": "base"})
    write_json(cache / "place_keys.json", [["City", i] for i in range(count)])
    write_json(cache / "provenance.json", {"teacher": "fixed"})
    np.save(cache / "paths.npy", np.array(paths))
    np.save(cache / "labels.npy", np.arange(count, dtype="int32"))
    np.save(cache / "train_features.npy", np.zeros((count, 4), dtype="float32"))
    ledger = night / "shards/s00000/state/results.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {"night_roots": [str(night)], "base_cache": str(cache), "input_dir": str(tmp_path / "inputs")}
    return config, rows, ledger


def test_snapshot_preserves_review_and_cache_labels_with_place_holdout(tmp_path):
    config, rows, ledger = fixture(tmp_path)
    # A repeated transport receipt is deduplicated, never a second training sample.
    with ledger.open("a") as stream:
        stream.write(json.dumps(rows[0]) + "\n")
    summary = build_snapshot(config)
    assert summary["counts"] == {"included": 20, "train": 19, "val": 1, "excluded": 1, "receipt_rows": 20}
    train, val = load_records(config, "train"), load_records(config, "val")
    assert {r["label"] for r in train}.isdisjoint({r["label"] for r in val})
    assert {r["source_index"] for r in train}.isdisjoint({r["source_index"] for r in val})
    assert all(r["source_index"] == r["label"] == r["place_id"] for r in train + val)
    assert all(r["status"] == "pending_verification" and r["passed"] is False and
               r["eligible_for_training"] is False for r in train + val)
    assert summary["excluded_reasons"] == {"duplicate_ledger_entry": 1}


def test_completed_snapshot_does_not_follow_new_generation(tmp_path):
    config, rows, ledger = fixture(tmp_path)
    original = build_snapshot(config)
    with ledger.open("a") as stream:
        stream.write('{"a_new_unfinished_result":')
    assert build_snapshot(config)["manifest_sha256"] == original["manifest_sha256"]
    assert len(load_records(config, "all")) == 20
    changed = {**config, "night_roots": [str(tmp_path)]}
    with pytest.raises(NightDataIntegrityError, match="configuration differs"):
        build_snapshot(changed)


def test_hash_and_label_conflicts_abort_instead_of_silently_skipping(tmp_path):
    config, rows, ledger = fixture(tmp_path, count=2)
    rows[0]["place_id"] = 1
    rows[0]["result_sha256"] = _fingerprint({k: v for k, v in rows[0].items() if k != "result_sha256"})
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(NightDataIntegrityError, match="label conflicts"):
        build_snapshot(config)
    rows[0]["place_id"] = 0
    rows[0]["output_sha256"] = "0" * 64
    rows[0]["result_sha256"] = _fingerprint({k: v for k, v in rows[0].items() if k != "result_sha256"})
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(NightDataIntegrityError, match="SHA256 differs"):
        build_snapshot(config)


def test_corrupt_matching_hash_image_and_torn_tail_are_explicit_exclusions(tmp_path):
    config, rows, ledger = fixture(tmp_path, count=3)
    output = Path(rows[0]["output_path"])
    output.write_bytes(b"not a PNG")
    rows[0]["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    rows[0]["result_sha256"] = _fingerprint({k: v for k, v in rows[0].items() if k != "result_sha256"})
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"unfinished":')
    summary = build_snapshot(config)
    assert summary["total"] == 2
    assert summary["excluded_reasons"] == {"incomplete_ledger_tail": 1, "undecodable_night_image": 1}


def test_frozen_image_mutation_is_detected(tmp_path):
    config, rows, _ = fixture(tmp_path, count=2)
    build_snapshot(config)
    Path(rows[0]["output_path"]).write_bytes(b"changed")
    with pytest.raises(NightDataIntegrityError, match="changed since"):
        load_records(config, "all")


def test_all_views_of_a_place_share_one_split_and_missing_output_is_excluded(tmp_path):
    config, rows, ledger = fixture(tmp_path)
    rows[1]["place_id"] = 0
    rows[1]["result_sha256"] = _fingerprint({k: v for k, v in rows[1].items() if k != "result_sha256"})
    labels = np.arange(20, dtype="int32")
    labels[1] = 0
    np.save(Path(config["base_cache"]) / "labels.npy", labels)
    Path(rows[-1]["output_path"]).unlink()
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = build_snapshot(config)
    records = load_records(config, "all")
    place_splits = {}
    for row in records:
        place_splits.setdefault(row["label"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in place_splits.values())
    assert len([row for row in records if row["label"] == 0]) == 2
    assert summary["excluded_reasons"] == {"missing_night_image": 1}
