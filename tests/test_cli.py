import torch
from conftest import make_table

from distrainer.block import write_block
from distrainer.cli import main
from distrainer.ledger import Ledger
from distrainer.log import BlockLog
from distrainer.trainer import CheckpointIO


def test_inspect_export_log_ls_gc(store, tmp_path, capsys):
    fs, root = store
    log = BlockLog.create(fs, root, W=2, seed=3)
    for s in range(3):
        log.append([write_block(fs, root, f"s{s}b{i}", make_table(1)) for i in range(2)])
    assert main(["log-ls", root, "-v"]) == 0
    out = capsys.readouterr().out
    assert "W=2 seed=3" in out and "segments: 3 (0..2)" in out and "ended: False" in out
    assert "00000001 pass=0 positions 2..3" in out
    assert main(["gc", root, "--keep-from", "1"]) == 0
    assert "deleted segments: [0]" in capsys.readouterr().out
    assert main(["log-ls", str(tmp_path / "nothing")]) == 1

    io = CheckpointIO(scratch_dir=str(tmp_path / "scratch"))
    ckpt = io.save(torch.nn.Linear(2, 1), None, Ledger(segment=1, cursor=1, world_size=2))
    assert main(["inspect", ckpt.path]) == 0
    out = capsys.readouterr().out
    assert "'segment': 1" in out and "done positions in segment 1: 2" in out
    assert main(["export", ckpt.path, str(tmp_path / "exported")]) == 0
    assert (tmp_path / "exported" / "ledger.json").exists()
