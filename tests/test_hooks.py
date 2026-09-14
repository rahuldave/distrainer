import pytest
from conftest import AppendNextSegments

from distrainer.config import DistrainerConfig
from distrainer.hooks import SegmentHook, build_hooks, hook_specs, load_entry


def test_hook_specs_accept_string_and_mapping_forms():
    specs = hook_specs(
        {
            "a": "conftest:AppendNextSegments",
            "b": {"entry": "conftest:AppendNextSegments", "segments": 3},
        }
    )
    assert specs == [
        ("a", "conftest:AppendNextSegments", {}),
        ("b", "conftest:AppendNextSegments", {"segments": 3}),
    ]
    assert hook_specs({}) == []
    for bad in (
        {"a": {}},
        {"a": {"segments": 1}},
        {"a": "no_colon"},
        {"a": 5},
        {"a": {"entry": ":x"}},
        {"a": {"entry": "mod:"}},
        {"a": None},
    ):
        with pytest.raises(ValueError):
            hook_specs(bad)


def test_load_entry_resolves_module_attr():
    assert load_entry("conftest:AppendNextSegments") is AppendNextSegments
    assert load_entry("conftest:AppendNextSegments.on_segment_end").__name__ == "on_segment_end"
    with pytest.raises(ModuleNotFoundError):
        load_entry("no.such.module:X")
    with pytest.raises(AttributeError):
        load_entry("conftest:Nope")


def test_build_hooks_passes_config_and_args_and_checks_the_protocol():
    cfg = DistrainerConfig.from_dict(
        {"hooks": {"h": {"entry": "conftest:AppendNextSegments", "segments": 5}}}
    )
    (hook,) = build_hooks(cfg)
    assert isinstance(hook, SegmentHook)
    assert hook.segments == 5 and hook.config is cfg
    assert build_hooks(DistrainerConfig.from_dict({})) == []
    with pytest.raises(TypeError, match="on_segment_end"):
        build_hooks(DistrainerConfig.from_dict({"hooks": {"h": "conftest:NotAHook"}}))
