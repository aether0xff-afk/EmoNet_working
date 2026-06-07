from __future__ import annotations

import json

from emonet_v7.run_logger import RunLogger


def test_run_logger_writes_jsonl_records(tmp_path, capsys) -> None:
    logger = RunLogger(output_dir=tmp_path, verbose=True)
    logger.section("demo")
    logger.log("metric", "측정 완료", value=0.5, tags=["a", "b"])

    captured = capsys.readouterr().out
    assert "[EmoNet][001][section]" in captured
    assert "[EmoNet][002][metric]" in captured
    assert "value=0.5" in captured

    records = [
        json.loads(line)
        for line in (tmp_path / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in records] == ["section", "metric"]
    assert records[1]["value"] == 0.5
    assert records[1]["tags"] == ["a", "b"]
