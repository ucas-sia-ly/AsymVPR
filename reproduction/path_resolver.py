"""Strict adapters from benchmark image IDs to existing local image files.

This module never changes image bytes. Standard formatted datasets are used
directly when available; raw downloader layouts require the explicit adapters
below. The caller should record which layout and image encoding were used.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def _unique_index(paths: Iterable[Path], key, label: str) -> dict:
    result = {}
    for path in paths:
        identifier = key(path)
        if identifier in result:
            raise ValueError(f"{label}: duplicate image identity {identifier!r}")
        result[identifier] = path
    return result


def _fields(name: str) -> list[str]:
    fields = Path(name).name.split("@")
    if len(fields) < 9:
        raise ValueError(f"Expected a benchmark filename with @ metadata: {name}")
    return fields


def resolve_paths(dataset: str, root: Path | str, paths: Sequence[str]) -> list[Path]:
    """Resolve every ID in order, rejecting missing, empty or duplicate images.

    ``nordland`` accepts the original VPR-Bench ``ref/`` and ``query/`` release.
    In the native downloader layout, references are winter files in
    ``images/test/queries`` and summer queries in ``images/test/database``.
    Official query ID i maps to native frame 10*i; reference ID i maps to i.
    """
    name = dataset.lower().replace("-", "").replace("_", "")
    aliases = {"tokyo": "tokyo247", "pittsburgh": "pitts250k", "mslsval": "msls"}
    name = aliases.get(name, name)
    if name not in {"tokyo247", "nordland", "amstertime", "pitts250k", "msls"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {root}")
    indexes = {}
    resolved = []
    for value in paths:
        identifier = str(value)
        if Path(identifier).is_absolute() or ".." in Path(identifier).parts:
            raise ValueError(f"Expected a relative benchmark image ID: {identifier}")
        direct = root / identifier
        if direct.is_file():
            path = direct
        elif name == "msls":
            path = root / identifier.removeprefix("train_val/")
        elif name == "pitts250k":
            path = root / "Pittsburgh" / identifier
        elif name == "tokyo247":
            fields = _fields(identifier)
            role = Path(identifier).parts[0]
            if role == "queries":
                raw_name = fields[7]
                if Path(raw_name).name != raw_name or not raw_name.endswith(".jpg"):
                    raise ValueError(f"Invalid Tokyo query image ID: {identifier}")
                path = root / "queries" / raw_name
            elif role == "database":
                yaw = int(fields[8])
                if not 0 <= yaw < 12:
                    raise ValueError(f"Invalid Tokyo yaw index: {identifier}")
                if "tokyo" not in indexes:
                    indexes["tokyo"] = _unique_index(
                        (root / "database_gsv_vga").rglob("*.png"),
                        lambda p: p.name, "Tokyo database")
                raw_name = f"{fields[7]}_012_{yaw * 30:03d}.png"
                try:
                    path = indexes["tokyo"][raw_name]
                except KeyError as exc:
                    raise FileNotFoundError(f"Tokyo image {identifier} -> {raw_name}") from exc
            else:
                raise ValueError(f"Invalid Tokyo role: {identifier}")
        elif name == "nordland":
            role = Path(identifier).parts[0]
            if role not in {"ref", "query"}:
                raise ValueError(f"Invalid Nordland role: {identifier}")
            native_role = "queries" if role == "ref" else "database"
            if native_role not in indexes:
                indexes[native_role] = _unique_index(
                    (root / "images/test" / native_role).glob("*.jpg"),
                    lambda p: int(_fields(str(p))[7]), f"Nordland {native_role}")
            frame = int(Path(identifier).stem) * (10 if role == "query" else 1)
            try:
                path = indexes[native_role][frame]
            except KeyError as exc:
                raise FileNotFoundError(f"Nordland image {identifier} -> {native_role} frame {frame}") from exc
        else:  # AmsterTime modern reference / historical query direction.
            fields = _fields(identifier)
            pair_id = fields[7]
            if not pair_id.isdigit():
                raise ValueError(f"Invalid AmsterTime pair ID: {identifier}")
            role = Path(identifier).parts[0]
            if role == "database":
                path = root / "new" / f"{pair_id}.png"
            elif role == "queries":
                path = root / "old" / f"{pair_id}.jpg"
            else:
                raise ValueError(f"Invalid AmsterTime role: {identifier}")
        path = path.resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{dataset}: missing/empty image {identifier} -> {path}")
        resolved.append(path)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{dataset}: multiple benchmark IDs resolve to the same image")
    return resolved


def validate_amstertime_pairs(root: Path | str, references: Sequence[str],
                             queries: Sequence[str]) -> dict:
    """Check raw CSV pairing against both complete benchmark ID lists."""
    root = Path(root)
    db_ids = [_fields(str(value))[7] for value in references]
    q_ids = [_fields(str(value))[7] for value in queries]
    with (root / "image_pairs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = [(f"{int(row['new_num']):04d}", f"{int(row['old_num']):04d}") for row in rows]
    if (len(set(db_ids)) != len(db_ids) or len(set(q_ids)) != len(q_ids)
            or len(pairs) != len(db_ids) or len(set(pairs)) != len(pairs)
            or any(modern != historical for modern, historical in pairs)
            or {modern for modern, _ in pairs} != set(db_ids)
            or {historical for _, historical in pairs} != set(q_ids)):
        raise ValueError("AmsterTime CSV pairs do not match the complete benchmark IDs")
    return {"csv_rows": len(rows), "reference_ids": len(db_ids),
            "query_ids": len(q_ids), "same_id_pairs": True,
            "references": "modern new/*.png", "queries": "historical old/*.jpg"}
