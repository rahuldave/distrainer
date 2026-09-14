import os

import pyarrow.fs as pafs
import pytest

from distrainer import storage
from distrainer.storage import StorageConfig, build_filesystem, resolve


def test_local_build_creates_absolute_root(tmp_path):
    fs, root = build_filesystem(StorageConfig(kind="local", path=str(tmp_path / "a" / "b")))
    assert isinstance(fs, pafs.LocalFileSystem)
    assert os.path.isabs(root) and os.path.isdir(root)


def test_local_relative_path_is_absolutised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, root = build_filesystem(StorageConfig(kind="local", path="rel"))
    assert root == str(tmp_path / "rel")


def test_from_dict_rejects_unknown_keys_and_bad_kind():
    with pytest.raises(ValueError):
        StorageConfig.from_dict({"kind": "local", "bogus": 1})
    with pytest.raises(ValueError):
        StorageConfig.from_dict({"kind": "gcs"})
    cfg = StorageConfig.from_dict(
        {"kind": "s3", "path": "bucket/prefix", "endpoint": "http://x:9000"}
    )
    assert cfg.asdict()["endpoint"] == "http://x:9000"


def test_s3_build_uses_endpoint_and_env_credentials(monkeypatch):
    monkeypatch.setenv("S3_ACCESS_KEY", "ak")
    monkeypatch.setenv("S3_SECRET_KEY", "sk")
    captured = {}

    class FakeS3(pafs.LocalFileSystem):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(storage.pafs, "S3FileSystem", FakeS3)
    fs, root = build_filesystem(
        StorageConfig(
            kind="s3", path="/bucket/prefix/", endpoint="http://minio:9000", region="auto"
        )
    )
    assert root == "bucket/prefix"
    assert captured == {
        "access_key": "ak",
        "secret_key": "sk",
        "endpoint_override": "minio:9000",
        "scheme": "http",
        "region": "auto",
    }


def test_s3_requires_bucket(monkeypatch):
    monkeypatch.setattr(storage.pafs, "S3FileSystem", lambda **kw: pafs.LocalFileSystem())
    with pytest.raises(ValueError):
        build_filesystem(StorageConfig(kind="s3", path="/"))


def test_resolve_local_and_file_uri(tmp_path):
    fs, root = resolve(str(tmp_path / "x"))
    assert isinstance(fs, pafs.LocalFileSystem) and root == str(tmp_path / "x")
    _, root2 = resolve(f"file://{tmp_path}/y")
    assert root2 == str(tmp_path / "y")


def test_helpers_roundtrip_and_atomic_write(store):
    fs, root = store
    d = storage.join(root, "sub")
    storage.ensure_dir(fs, d)
    p = storage.join(d, "f.json")
    assert not storage.exists(fs, p)
    storage.write_atomic(fs, p, b'{"a": 1}')
    assert storage.exists(fs, p)
    assert storage.read_bytes(fs, p) == b'{"a": 1}'
    assert storage.list_names(fs, d) == ["f.json"]  # no leftover temp file
    storage.write_atomic(fs, p, b"2")  # overwrite is allowed and atomic
    assert storage.read_bytes(fs, p) == b"2"
    storage.delete(fs, p)
    storage.delete(fs, p)  # missing_ok
    assert storage.list_names(fs, d) == []
    assert storage.list_names(fs, storage.join(root, "nope")) == []
