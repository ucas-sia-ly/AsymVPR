from pathlib import Path

import pytest

from reproduction.path_resolver import resolve_paths, validate_amstertime_pairs


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image fixture")
    return path.resolve()


def test_tokyo_tile_identity_and_query_order(tmp_path):
    tile = image(tmp_path / "database_gsv_vga/03814/381418.22_3946450.43/pano_id_012_330.png")
    query = image(tmp_path / "queries/00685.jpg")
    identifiers = ["queries/@0381523.71@3946790.15@54@S@035.65786@0139.69112@00685.jpg@@@@@@@@.jpg",
                   "database/@0381418.22@3946450.43@54@S@035.65478@0139.69001@pano_id@11@@@@@@@.jpg"]
    assert resolve_paths("tokyo247", tmp_path, identifiers) == [query, tile]


def test_tokyo_duplicate_raw_identity_is_rejected(tmp_path):
    image(tmp_path / "database_gsv_vga/a/pano_012_000.png")
    image(tmp_path / "database_gsv_vga/b/pano_012_000.png")
    with pytest.raises(ValueError, match="duplicate image identity"):
        resolve_paths("tokyo247", tmp_path, ["database/@0@0@54@S@0@0@pano@00@@@@@@@.jpg"])


def test_nordland_native_season_direction_and_stride(tmp_path):
    winter = image(tmp_path / "images/test/queries/@0@23.8@@@@@10@@@@@@@@.jpg")
    summer = image(tmp_path / "images/test/database/@0@23.8@@@@@10@@@@@@@@.jpg")
    assert resolve_paths("nordland", tmp_path, ["query/0000001.jpg", "ref/0000010.jpg"]) == [summer, winter]
    official = image(tmp_path / "query/0000001.jpg")
    assert resolve_paths("nordland", tmp_path, ["query/0000001.jpg"]) == [official]


def test_amstertime_modern_reference_and_csv_pairing(tmp_path):
    modern = image(tmp_path / "new/0002.png")
    historical = image(tmp_path / "old/0002.jpg")
    references = ["database/@0@2000@@@@@0002@@@@@@@@.jpg"]
    queries = ["queries/@0@2000@@@@@0002@@@@@@@@.jpg"]
    assert resolve_paths("amstertime", tmp_path, references + queries) == [modern, historical]
    csv_path = tmp_path / "image_pairs.csv"
    csv_path.write_text("new_num,old_num\n2,2\n")
    assert validate_amstertime_pairs(tmp_path, references, queries)["same_id_pairs"]
    csv_path.write_text("new_num,old_num\n2,3\n")
    with pytest.raises(ValueError, match="CSV pairs"):
        validate_amstertime_pairs(tmp_path, references, queries)


def test_missing_and_reused_images_are_not_silently_accepted(tmp_path):
    image(tmp_path / "city/query/images/x.jpg")
    with pytest.raises(FileNotFoundError, match="missing/empty"):
        resolve_paths("msls", tmp_path, ["train_val/city/query/images/missing.jpg"])
    with pytest.raises(ValueError, match="same image"):
        resolve_paths("msls", tmp_path, ["train_val/city/query/images/x.jpg", "city/query/images/x.jpg"])
