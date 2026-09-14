import torch

from distrainer.ledger import Ledger
from distrainer.trainer import CheckpointIO, TrainInfo, checkpoint_dir_name, lanes_from, unwrap


def test_checkpoint_roundtrip_and_metadata(tmp_path):
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ledger = Ledger(segment=2, cursor=3, world_size=3, pass_idx=1, run_attempt=1)
    io = CheckpointIO(scratch_dir=str(tmp_path / "scratch"), run_name="t")
    ckpt = io.save(model, opt, ledger, extra={"note": "x"})
    assert checkpoint_dir_name(ledger) == "checkpoint_g000002_s0003"
    assert CheckpointIO.read_ledger(ckpt) == ledger
    assert ckpt.get_metadata()["note"] == "x"
    fresh = torch.nn.Linear(3, 1)
    fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
    assert CheckpointIO.load(ckpt, fresh, fresh_opt) == ledger
    assert torch.equal(fresh.weight, model.weight)
    assert CheckpointIO.load(ckpt) == ledger  # ledger only
    io.cleanup()
    assert not (tmp_path / "scratch").exists() or not list(io_dir_children(tmp_path / "scratch"))


def io_dir_children(p):
    return list(p.rglob("*")) if p.exists() else []


def test_read_ledger_falls_back_to_file_without_metadata(tmp_path):
    from ray.train import Checkpoint

    d = tmp_path / "ck"
    d.mkdir()
    Ledger(segment=1, cursor=1, world_size=2).save(str(d))
    assert CheckpointIO.read_ledger(Checkpoint.from_directory(str(d))) == Ledger(1, 1, 2)


def test_unwrap_and_train_info_defaults():
    from distrainer.config import DistrainerConfig

    m = torch.nn.Linear(1, 1)

    class Wrapped:
        module = m

    assert unwrap(Wrapped()) is m and unwrap(m) is m
    info = TrainInfo(rank=1, world_size=2, config=DistrainerConfig.from_dict({"train": {"lr": 1}}))
    assert info.train == {"lr": 1} and info.position == 0 and info.device.type == "cpu"


def test_lanes_from_resumes_and_stops(store):
    from conftest import make_table

    from distrainer.block import write_block
    from distrainer.log import BlockLog

    fs, root = store
    log = BlockLog.create(fs, root, W=4, seed=1)
    for s in range(2):
        log.append([write_block(fs, root, f"s{s}b{i}", make_table(1)) for i in range(4)])
    log.end()
    gen = lanes_from(log, rank=1, world_size=2, seq=0, start_step=1, poll_s=0.01)
    lanes = list(gen(None))
    assert [(seg.seq, [p for p, _ in ln]) for seg, ln in lanes] == [(0, [3]), (1, [5, 7])]


def test_metric_aggregator_means_numeric_metrics_only():
    from distrainer.trainer import MetricAggregator

    agg = MetricAggregator()
    agg.add({"loss": 1.0, "acc": 0.5, "name": "x", "flag": True})
    agg.add({"loss": 3.0, "acc": 1.0})
    out = agg.flush({"loss": 3.0, "position": 7})
    assert out == {
        "loss": 3.0,
        "position": 7,
        "loss_mean": 2.0,
        "acc_mean": 0.75,
        "steps_in_report": 2,
    }
    assert agg.flush({"loss": 9.0}) == {"loss": 9.0}  # window reset


def test_config_report_and_poll_fields():
    from distrainer.config import DistrainerConfig

    cfg = DistrainerConfig.from_dict({})
    assert cfg.checkpoint.report_every_step is False and cfg.ray_health_check_interval_s == 0.5
    import pytest

    with pytest.raises(ValueError):
        DistrainerConfig.from_dict({"ray_health_check_interval_s": 0})
