"""The cluster scenario runner's process bookkeeping (the scenarios themselves need docker)."""

import subprocess
import sys

from integration_tests.cluster import run_scenarios as rs


def test_finish_returns_the_process_output_and_final_metrics_parses_it(tmp_path):
    log_path = tmp_path / "p.log"
    log = open(log_path, "w")  # noqa: SIM115 (finish closes it)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "print('hello'); print(\"final metrics: {'reports': 3, 'x': 1.5}\")",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rs._logs[proc.pid] = (log_path, log)
    rs._running.append(proc)
    out = rs.finish(proc, name="p")
    assert "hello" in out and proc not in rs._running and proc.pid not in rs._logs
    assert log.closed
    assert rs.final_metrics(out) == {"reports": 3, "x": 1.5}
    try:
        rs.final_metrics("nothing here")
    except RuntimeError as exc:
        assert "final metrics" in str(exc)
    else:
        raise AssertionError("final_metrics must fail without the line")
