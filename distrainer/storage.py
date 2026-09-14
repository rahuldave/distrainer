"""distrainer.storage: one ``(pyarrow filesystem, root)`` pair for blocks, log, checkpoints, audit.

Spec section 6.3. ``kind: local`` (a directory) is the default; ``kind: s3`` covers AWS S3 and
S3-compatible stores such as MinIO or R2 through an explicit endpoint. Every other module takes
the ``(fs, root)`` pair produced here and joins relative paths onto ``root`` with :func:`join`.
"""

from __future__ import annotations

import os
import posixpath
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import pyarrow.fs as pafs

Kind = Literal["local", "s3"]


@dataclass(frozen=True)
class StorageConfig:
    """Where a store lives. ``path`` is a directory for ``local`` and ``bucket/prefix`` for
    ``s3``."""

    kind: Kind = "local"
    path: str = "runs"
    endpoint: str | None = None
    region: str | None = None
    access_key_env: str = "S3_ACCESS_KEY"
    secret_key_env: str = "S3_SECRET_KEY"
    anonymous: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StorageConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown storage keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.kind not in ("local", "s3"):
            raise ValueError(f"storage.kind must be 'local' or 's3', got {cfg.kind!r}")
        return cfg

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def build_filesystem(cfg: StorageConfig, create: bool = True) -> tuple[pafs.FileSystem, str]:
    """Return ``(fs, root)``. Local roots are absolute paths, created if ``create`` is set."""
    if cfg.kind == "local":
        root = os.path.abspath(os.path.expanduser(cfg.path))
        fs = pafs.LocalFileSystem()
        if create:
            fs.create_dir(root, recursive=True)
        return fs, root
    if cfg.kind == "s3":
        kwargs: dict[str, Any] = {}
        if cfg.anonymous:
            kwargs["anonymous"] = True
        else:
            access_key = os.environ.get(cfg.access_key_env)
            secret_key = os.environ.get(cfg.secret_key_env)
            if access_key and secret_key:
                kwargs["access_key"] = access_key
                kwargs["secret_key"] = secret_key
        if cfg.endpoint:
            parsed = urlparse(cfg.endpoint if "://" in cfg.endpoint else f"https://{cfg.endpoint}")
            kwargs["endpoint_override"] = parsed.netloc
            kwargs["scheme"] = parsed.scheme
        if cfg.region:
            kwargs["region"] = cfg.region
        fs = pafs.S3FileSystem(**kwargs)
        root = cfg.path.strip("/")
        if not root:
            raise ValueError("storage.path must name at least a bucket for kind 's3'")
        return fs, root
    raise ValueError(f"unsupported storage kind {cfg.kind!r}")


def resolve(uri: str, create: bool = True, **s3_options: Any) -> tuple[pafs.FileSystem, str]:
    """``(fs, root)`` for ``s3://bucket/prefix``, ``file:///dir`` or a plain path.

    Other URI schemes (``gs://``, ``abfs://`` ...) are rejected rather than treated as local
    directories; pass an explicit ``pyarrow.fs.FileSystem`` to the callers that accept one.
    """
    if uri.startswith("s3://"):
        cfg = StorageConfig(kind="s3", path=uri[len("s3://") :], **s3_options)
        return build_filesystem(cfg, create=create)
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    elif "://" in uri:
        raise ValueError(f"unsupported URI scheme in {uri!r}; use s3://, file:// or a plain path")
    return build_filesystem(StorageConfig(kind="local", path=uri), create=create)


def s3_options_from_env() -> dict[str, Any]:
    """``resolve``/``StorageConfig`` keyword arguments for an S3-compatible endpoint set through
    ``S3_ENDPOINT`` (and ``S3_REGION``); empty when the environment names no endpoint."""
    endpoint = os.environ.get("S3_ENDPOINT")
    if not endpoint:
        return {}
    return {"endpoint": endpoint, "region": os.environ.get("S3_REGION", "auto")}


# ---- small filesystem helpers shared by log, block, audit, checkpoint code ----


def join(root: str, *parts: str) -> str:
    return posixpath.join(root, *parts)


def exists(fs: pafs.FileSystem, path: str) -> bool:
    return fs.get_file_info(path).type != pafs.FileType.NotFound


def ensure_dir(fs: pafs.FileSystem, path: str) -> None:
    fs.create_dir(path, recursive=True)


def read_bytes(fs: pafs.FileSystem, path: str) -> bytes:
    with fs.open_input_stream(path) as f:
        return f.read()


def write_bytes(fs: pafs.FileSystem, path: str, data: bytes) -> None:
    with fs.open_output_stream(path) as f:
        f.write(data)


OBJECT_STORE_TYPES = frozenset({"s3", "gcs", "abfs", "azure"})


def local_path(fs: pafs.FileSystem, path: str) -> str | None:
    """The OS path behind ``path`` if ``fs`` is a (subtree of a) local filesystem, else None."""
    while isinstance(fs, pafs.SubTreeFileSystem):
        path = posixpath.join(fs.base_path, path)
        fs = fs.base_fs
    return path if isinstance(fs, pafs.LocalFileSystem) else None


def is_local(fs: pafs.FileSystem) -> bool:
    return local_path(fs, "") is not None


def write_atomic(fs: pafs.FileSystem, path: str, data: bytes) -> None:
    """Readers see either nothing or the whole file.

    Local (including subtree views): write ``<path>.tmp-<uuid>`` next to the target, then
    ``os.replace``. Object stores: a single ``put`` is already atomic. Any other filesystem:
    temp file plus ``fs.move``. Leftover ``.tmp-*`` files are orphans and are ignored.
    """
    target = local_path(fs, path)
    if target is not None:
        tmp = f"{target}.tmp-{uuid.uuid4().hex}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    elif fs.type_name in OBJECT_STORE_TYPES:
        write_bytes(fs, path, data)
    else:
        tmp = f"{path}.tmp-{uuid.uuid4().hex}"
        write_bytes(fs, tmp, data)
        fs.move(tmp, path)


def delete(fs: pafs.FileSystem, path: str, missing_ok: bool = True) -> None:
    if missing_ok and not exists(fs, path):
        return
    fs.delete_file(path)


def list_names(fs: pafs.FileSystem, path: str) -> list[str]:
    """Base names of the files directly under ``path`` (empty if it does not exist)."""
    if not exists(fs, path):
        return []
    selector = pafs.FileSelector(path, recursive=False, allow_not_found=True)
    return sorted(
        posixpath.basename(info.path)
        for info in fs.get_file_info(selector)
        if info.type == pafs.FileType.File
    )
