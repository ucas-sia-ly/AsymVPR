"""Report provenance regressions; every input/output lives in a temp directory."""
import json

import pytest

from reproduction import report


@pytest.fixture
def isolated_report(tmp_path, monkeypatch):
    # generate() intentionally reads this project's historical source record.
    # Redirect its project root as well as the supplied summary/output paths.
    monkeypatch.setattr(report, '__file__', str(tmp_path / 'reproduction' / 'report.py'))
    results = tmp_path / 'results'
    results.mkdir()
    (results / 'training_history.json').write_text(json.dumps({
        'best_checkpoint': {'sha256': 'historical-best'},
    }))
    evaluation = tmp_path / 'evaluation'
    evaluation.mkdir()
    return evaluation, tmp_path / 'REPORT.md'


def result(metrics, paper, protocol, note=None):
    value = dict(metrics_percent=metrics, paper_table1_percent=paper,
                 references={'count': 100}, queries={'count': 20}, protocol=protocol)
    if note is not None:
        value['paper_comparison_note'] = note
    return value


def test_execution_hash_mismatch_does_not_publish_stale_summary_scores(isolated_report):
    evaluation, output = isolated_report
    (evaluation / 'summary.json').write_text(json.dumps({
        'checkpoint_sha256': 'previous-run-checkpoint',
        'datasets': {'msls': result({'R@1': 77.1234, 'R@5': 88.5678},
                                    {'R@1': 92.3, 'R@5': 95.9}, {'split': 'validation'})},
    }))
    (evaluation / 'execution.json').write_text(json.dumps({
        'checkpoint_sha256': 'current-run-checkpoint', 'status': 'interrupted',
    }))
    report.generate(evaluation / 'summary.json', output)
    text = output.read_text()
    assert '| MSLS val | 尚未产生 |' in text
    assert '77.1234' not in text
    assert '88.5678' not in text
    assert '已排除属于其他权重的旧汇总' in text
    assert '**interrupted**' in text


def test_nordland_salad_protocol_has_no_ordinary_paper_delta(isolated_report):
    evaluation, output = isolated_report
    note = 'Ground-truth tolerance differs from paper; difference is diagnostic only'
    (evaluation / 'summary.json').write_text(json.dumps({
        'checkpoint_sha256': 'historical-best',
        'datasets': {'nordland': result({'R@1': 80., 'R@5': 90.},
            {'R@1': 74.6, 'R@5': 88.3},
            {'protocol_variant': 'salad', 'positives': 'published SALAD ground truth +/- 1'}, note)},
    }))
    report.generate(evaluation / 'summary.json', output)
    text = output.read_text()
    nordland_row = next(line for line in text.splitlines() if line.startswith('| Nordland |'))
    assert '80.0000' in nordland_row
    assert '—（协议不同）' in nordland_row
    assert '+5.4000' not in nordland_row
    assert note in text
