import textwrap

import pyarrow.fs as pafs
import pytest

from distrainer.config import DistrainerConfig, load_config
from distrainer.policy import Any as AnyPolicy
from distrainer.policy import build_policy

SPEC_YAML = """
run_name: toy
storage_path: {runs}
store_root: {blocks}
seed: 1234
log:
  W: 24
  passes: 2
  wait_poll_s: 1.0
  retention_segments: 4
  shuffle_buffer_segments: 1
checkpoint:
  policy: any
  every_k: 4
  time_budget_s: null
  num_to_keep: 3
loader:
  prefetch: 2
  threads: 2
scaling:
  num_workers: [2, 4]
  resources_per_worker: {{CPU: 1}}
  use_gpu: false
  elastic_resize_monitor_interval_s: 15
failure:
  max_failures: 3
hooks:
  remine: {{entry: "examples.toy_contrastive.remine:RemineHook", segments: 8}}
"""


def test_spec_example_loads_and_builds_filesystems(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        textwrap.dedent(SPEC_YAML.format(runs=tmp_path / "runs", blocks=tmp_path / "blocks"))
    )
    cfg = load_config(str(path))
    assert cfg.run_name == "toy" and cfg.seed == 1234
    assert cfg.scaling.num_workers == (2, 4) and cfg.scaling.elastic
    assert (cfg.scaling.min_workers, cfg.scaling.max_workers) == (2, 4)
    assert list(cfg.allowed_world_sizes()) == [2, 3, 4]
    assert cfg.hooks == {
        "remine": {"entry": "examples.toy_contrastive.remine:RemineHook", "segments": 8}
    }
    assert cfg.checkpoint.every_k == 4 and cfg.checkpoint.time_budget_s is None
    fs, root = cfg.store_fs()
    assert isinstance(fs, pafs.LocalFileSystem) and root == str(tmp_path / "blocks")
    _, runs_root = cfg.runs_fs()
    assert runs_root == str(tmp_path / "runs")
    assert isinstance(build_policy(cfg.checkpoint.as_policy_dict()), AnyPolicy)
    assert DistrainerConfig.from_dict(cfg.asdict()).scaling.num_workers == (2, 4)


def test_defaults_are_valid():
    cfg = DistrainerConfig.from_dict({})
    assert cfg.log.W == 16 and cfg.scaling.num_workers == 2 and not cfg.scaling.elastic
    assert cfg.storage.kind == "local"
    assert cfg.log.gc is False and cfg.log.gc_blocks is True and cfg.log.retention_segments == 4
    assert cfg.hooks == {}


def test_s3_storage_section_uses_paths_from_top_level(monkeypatch):
    captured = {}

    class FakeS3(pafs.LocalFileSystem):
        def __init__(self, **kw):
            captured.update(kw)
            super().__init__()

    monkeypatch.setattr("distrainer.storage.pafs.S3FileSystem", FakeS3)
    cfg = DistrainerConfig.from_dict(
        {
            "storage_path": "distrainer/runs",
            "store_root": "distrainer/blocks",
            "storage": {"kind": "s3", "endpoint": "http://minio:9000", "anonymous": True},
        }
    )
    _, root = cfg.store_fs()
    assert root == "distrainer/blocks" and captured["endpoint_override"] == "minio:9000"
    _, runs = cfg.runs_fs()
    assert runs == "distrainer/runs"


@pytest.mark.parametrize(
    "bad",
    [
        {"log": {"W": 12}, "scaling": {"num_workers": [2, 5]}},  # lcm(2,3,4,5)=60
        {"log": {"W": 15}},
        {"scaling": {"num_workers": [3, 2]}},
        {"scaling": {"num_workers": 0}},
        {"checkpoint": {"policy": "weekly"}},
        {"checkpoint": {"policy": "time", "time_budget_s": None}},
        {"checkpoint": {"policy": "every_k", "every_k": None}},
        {"storage": {"kind": "local", "path": "elsewhere"}},
        {"checkpoint": {"every_k": 0}},
        {"loader": {"prefetch": 0}},
        {"run_name": "a/b"},
        {"bogus": 1},
        {"log": {"window": 16}},
        {"hooks": {"remine": {"every_segment": True}}},  # no entry
        {"hooks": {"remine": "not-an-entry"}},
        {"hooks": ["examples.toy_contrastive.remine:RemineHook"]},  # a list, not a mapping
        {"log": {"gc": True, "retention_segments": 0}},  # the controller may restart one behind
    ],
)
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        DistrainerConfig.from_dict(bad)


def test_w_must_cover_every_allowed_world_size():
    cfg = DistrainerConfig.from_dict({"log": {"W": 12}, "scaling": {"num_workers": [2, 4]}})
    assert list(cfg.allowed_world_sizes()) == [2, 3, 4]
    with pytest.raises(ValueError):
        DistrainerConfig.from_dict({"log": {"W": 8}, "scaling": {"num_workers": [2, 3]}})


def test_asdict_yaml_roundtrip_and_direct_construction_validates(tmp_path):
    import yaml

    from distrainer.config import DistrainerConfig, LogConfig

    cfg = DistrainerConfig.from_dict({"scaling": {"num_workers": [2, 4]}, "log": {"W": 24}})
    text = yaml.safe_dump(cfg.asdict())
    again = DistrainerConfig.from_dict(yaml.safe_load(text))
    assert again.scaling.num_workers == (2, 4) and again.log.W == 24
    with pytest.raises(ValueError):
        DistrainerConfig(log=LogConfig(W=0))


def test_overrides_are_dotted_yaml_values(tmp_path):
    from distrainer.config import apply_overrides

    d = apply_overrides(
        {"log": {"W": 12}}, ["log.W=24", "checkpoint.num_to_keep=null", "run_name=x"]
    )
    assert d == {"log": {"W": 24}, "checkpoint": {"num_to_keep": None}, "run_name": "x"}
    path = tmp_path / "c.yaml"
    path.write_text("log:\n  W: 12\n")
    cfg = load_config(str(path), overrides=["scaling.num_workers=[2, 3]"])
    assert cfg.scaling.num_workers == (2, 3)
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])
    with pytest.raises(ValueError):
        apply_overrides({"run_name": "x"}, ["run_name.sub=1"])


# ---- M9: the parallel: section ----


def test_parallel_section_defaults_validates_and_round_trips():
    cfg = DistrainerConfig.from_dict({})
    assert cfg.parallel.kind == "ddp"
    cfg2 = DistrainerConfig.from_dict({"parallel": {"kind": "none"}})
    assert cfg2.parallel.kind == "none"
    assert DistrainerConfig.from_dict(cfg2.asdict()).parallel.kind == "none"
    with pytest.raises(ValueError, match="parallel.kind"):
        DistrainerConfig.from_dict({"parallel": {"kind": "tensor"}})
    with pytest.raises(ValueError, match="unknown keys in parallel"):
        DistrainerConfig.from_dict({"parallel": {"kind": "ddp", "buckets": 3}})


def test_parallel_kinds_and_the_outer_optimizer_fields():
    for kind in ("ddp", "none", "local_sgd", "diloco"):
        assert DistrainerConfig.from_dict({"parallel": {"kind": kind}}).parallel.kind == kind
    cfg = DistrainerConfig.from_dict({"parallel": {"kind": "diloco", "outer_lr": 0.5}})
    assert cfg.parallel.outer_lr == 0.5 and cfg.parallel.outer_momentum == 0.9
    with pytest.raises(ValueError, match="outer_lr"):
        DistrainerConfig.from_dict({"parallel": {"kind": "diloco", "outer_lr": 0}})
    with pytest.raises(ValueError, match="outer_momentum"):
        DistrainerConfig.from_dict({"parallel": {"kind": "diloco", "outer_momentum": 1.0}})
