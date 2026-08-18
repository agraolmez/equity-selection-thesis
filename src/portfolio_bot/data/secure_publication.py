"""Hardened, generic primitives for publishing immutable file bundles.

This module is intentionally independent of every Phase 2 compiler and
verifier.  It supplies only filesystem mechanics:

* an explicit directory is opened as a trusted root without walking from
  ``/``;
* every descendant directory is traversed fd-relative with ``O_NOFOLLOW``;
* staging files are created exclusively and are fsynced before use;
* immutable directories are validated as exact, hash-bound closed sets and
  atomically renamed into place;
* directory entries are fsynced in dependency order; and
* a mutable authority pointer is replaced last with atomic no-replace/exchange
  compare-and-swap semantics and authenticated rollback state; and
* obsolete owned entries are moved to explicit ``.retired`` recovery names,
  never deleted through a check-then-unlink race.

The public API deliberately accepts callbacks for format-specific validation.
For example, a CRSP compiler can wrap the binary stream returned by
``open_exclusive_binary`` in ``pyarrow.PythonFile`` and a validator can inspect
Parquet metadata through the read-only descriptor passed to it.  Callbacks
must neither close nor retain the supplied descriptor after returning.

Only flat closed bundles are supported: each member is one safe basename.
This keeps the security boundary small and matches content-addressed bundles
whose partitioning is represented by deterministic filenames.

Retired entries intentionally remain as recovery evidence.  Reclaiming them
requires a later authenticated retention policy with exclusive namespace
ownership; this module deliberately does not pretend POSIX offers an atomic
``unlink-if-inode-matches`` operation.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import hashlib
import io
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterator, Mapping, Sequence


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HASH_BLOCK_SIZE = 8 << 20


class SecurePublicationError(RuntimeError):
    """Raised when publication cannot be proven confined and durable."""


@dataclasses.dataclass(frozen=True)
class FileRecord:
    """Authenticated identity of one closed-bundle member."""

    name: str
    sha256: str
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class BundleSnapshot:
    """Deterministic file inventory for an exact closed directory."""

    files: tuple[FileRecord, ...]

    @property
    def hashes(self) -> Mapping[str, str]:
        """Return an immutable basename-to-SHA-256 mapping."""

        return MappingProxyType({record.name: record.sha256 for record in self.files})

    @property
    def sizes(self) -> Mapping[str, int]:
        """Return an immutable basename-to-size mapping."""

        return MappingProxyType(
            {record.name: record.size_bytes for record in self.files}
        )


@dataclasses.dataclass(frozen=True)
class PublishedBundle:
    """Result of an immutable-directory publication."""

    path: Path
    snapshot: BundleSnapshot
    reused_existing: bool


FileValidator = Callable[[str, int, FileRecord], None]
BundleValidator = Callable[[Mapping[str, FileRecord]], None]
PayloadValidator = Callable[[bytes], None]
Gate = Callable[[], None]


@dataclasses.dataclass
class _DirectoryAnchor:
    path: Path
    trusted_root: Path
    label: str
    descriptor: int
    opened: os.stat_result
    root_descriptor: int
    root_opened: os.stat_result
    closed: bool = False


@dataclasses.dataclass(frozen=True)
class _ObservedFile:
    record: FileRecord
    stat: os.stat_result


def _lexical_absolute(path: str | Path) -> Path:
    """Return an absolute path without resolving or following symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _safe_component(value: object, label: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or any(ord(character) < 32 or ord(character) == 127 for character in component)
    ):
        raise SecurePublicationError(f"Unsafe {label} component: {component!r}")
    return component


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_stable_regular(
    left: os.stat_result,
    right: os.stat_result,
    *,
    expected_nlink: int,
) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_nlink == expected_nlink
        and right.st_nlink == expected_nlink
        and all(getattr(left, field) == getattr(right, field) for field in fields)
    )


def _same_stable_file(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_stable_regular(left, right, expected_nlink=1)


def _same_stable_file_except_ctime(
    left: os.stat_result, right: os.stat_result
) -> bool:
    """Match an anchored regular file while permitting only ctime drift.

    This predicate is not sufficient authentication on its own.  Its sole
    caller immediately reopens and rehashes the same inode, then requires the
    previously authenticated SHA-256 before accepting the refreshed ctime.
    """

    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
    )
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_nlink == 1
        and right.st_nlink == 1
        and all(getattr(left, field) == getattr(right, field) for field in fields)
    )


def _same_stable_directory(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and all(getattr(left, field) == getattr(right, field) for field in fields)
    )


def _same_stable_directory_except_ctime(
    left: os.stat_result, right: os.stat_result
) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
    )
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and all(getattr(left, field) == getattr(right, field) for field in fields)
    )


def _require_regular_links(
    observed: os.stat_result,
    label: str,
    *,
    expected_nlink: int = 1,
) -> None:
    if stat.S_ISLNK(observed.st_mode):
        raise SecurePublicationError(f"Symlink is prohibited for {label}")
    if not stat.S_ISREG(observed.st_mode):
        raise SecurePublicationError(f"Anchored {label} is not a regular file")
    if observed.st_nlink != expected_nlink:
        raise SecurePublicationError(
            f"Anchored {label} has prohibited hard-link count "
            f"{observed.st_nlink}; expected {expected_nlink}"
        )


def _same_regular_inode(
    left: os.stat_result,
    right: os.stat_result,
    *,
    expected_nlink: int = 1,
) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_nlink == expected_nlink
        and right.st_nlink == expected_nlink
        and _same_inode(left, right)
    )


def _native_rename_at(
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
    *,
    operation: str,
) -> None:
    """Call an atomic native rename variant, or fail closed."""

    source_bytes = os.fsencode(_safe_component(source, "rename source"))
    destination_bytes = os.fsencode(
        _safe_component(destination, "rename destination")
    )
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flags = {"noreplace": 0x00000004, "exchange": 0x00000002}
    elif sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flags = {"noreplace": 0x00000001, "exchange": 0x00000002}
    else:
        function = None
        flags = {}
    if function is None or operation not in flags:
        raise SecurePublicationError(
            f"Atomic {operation} rename is unsupported on {sys.platform}"
        )
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        source_directory_fd,
        source_bytes,
        destination_directory_fd,
        destination_bytes,
        flags[operation],
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number))


def _rename_noreplace_at(
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
) -> None:
    """Atomically move an entry without replacing the destination."""

    _native_rename_at(
        source_directory_fd,
        source,
        destination_directory_fd,
        destination,
        operation="noreplace",
    )


def _rename_directory_noreplace_at(
    directory_fd: int, source: str, destination: str
) -> None:
    """Atomically move a sibling directory without replacing any entry."""

    _rename_noreplace_at(directory_fd, source, directory_fd, destination)


def _rename_exchange_at(directory_fd: int, left: str, right: str) -> None:
    """Atomically exchange two sibling entries, or fail closed."""

    _native_rename_at(
        directory_fd,
        left,
        directory_fd,
        right,
        operation="exchange",
    )


def _fsync_file(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SecurePublicationError(f"Cannot fsync {label}") from exc


def _fsync_directory(descriptor: int, label: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SecurePublicationError(f"Cannot fsync {label} directory") from exc


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_trusted_root_fd(
    trusted_root: Path, label: str
) -> tuple[int, os.stat_result]:
    """Open an explicit trust anchor directly, without walking from ``/``."""

    root = _lexical_absolute(trusted_root)
    if not root.is_absolute() or root.anchor != os.sep:
        raise SecurePublicationError(f"{label} trusted root must be absolute")
    descriptor: int | None = None
    try:
        before = os.stat(root, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise SecurePublicationError(
                f"Symlink is prohibited for {label} trusted root: {root}"
            )
        descriptor = os.open(root, _directory_flags())
        opened = os.fstat(descriptor)
        after = os.stat(root, follow_symlinks=False)
    except SecurePublicationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SecurePublicationError(
            f"Cannot open {label} trusted root: {root}"
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(before, opened)
        or not _same_inode(opened, after)
    ):
        os.close(descriptor)
        raise SecurePublicationError(f"{label} trusted-root identity changed")
    return descriptor, opened


def _assert_trusted_root_identity(
    trusted_root: Path, expected: os.stat_result, label: str
) -> None:
    descriptor, observed = _open_trusted_root_fd(trusted_root, label)
    try:
        if not _same_inode(expected, observed):
            raise SecurePublicationError(f"{label} trusted-root identity changed")
    finally:
        os.close(descriptor)


def _relative_beneath(path: Path, trusted_root: Path, label: str) -> Path:
    absolute = _lexical_absolute(path)
    root = _lexical_absolute(trusted_root)
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise SecurePublicationError(f"{label} escapes trusted root: {root}") from exc
    return relative


def _open_directory_fd_secure(
    path: Path,
    label: str,
    *,
    create: bool,
    trusted_root: Path,
) -> tuple[int, os.stat_result, int, os.stat_result]:
    """Traverse below ``trusted_root`` fd-relative without following links."""

    absolute = _lexical_absolute(path)
    root = _lexical_absolute(trusted_root)
    relative = _relative_beneath(absolute, root, label)
    current_fd, root_opened = _open_trusted_root_fd(root, label)
    retained_root_fd: int | None = None
    traversed = root
    try:
        retained_root_fd = os.dup(current_fd)
        if not _same_inode(root_opened, os.fstat(retained_root_fd)):
            raise SecurePublicationError(
                f"{label} retained trusted-root identity changed"
            )
        for raw_part in relative.parts:
            part = _safe_component(raw_part, f"{label} path")
            traversed = traversed / part
            created = False
            try:
                before = os.stat(
                    part, dir_fd=current_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                if not create:
                    raise SecurePublicationError(
                        f"Cannot open {label}: {absolute}"
                    )
                try:
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise SecurePublicationError(
                        f"Cannot create {label}: {traversed}"
                    ) from exc
                else:
                    created = True
                    _fsync_directory(current_fd, f"new {label} parent")
                try:
                    before = os.stat(
                        part, dir_fd=current_fd, follow_symlinks=False
                    )
                except OSError as exc:
                    raise SecurePublicationError(
                        f"Cannot authenticate created {label}: {traversed}"
                    ) from exc
            except OSError as exc:
                raise SecurePublicationError(
                    f"Cannot inspect {label}: {traversed}"
                ) from exc
            if stat.S_ISLNK(before.st_mode):
                raise SecurePublicationError(
                    f"Symlink is prohibited in {label} path: {traversed}"
                )
            child_fd: int | None = None
            try:
                child_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
                opened = os.fstat(child_fd)
                after = os.stat(
                    part, dir_fd=current_fd, follow_symlinks=False
                )
            except OSError as exc:
                if child_fd is not None:
                    os.close(child_fd)
                raise SecurePublicationError(
                    f"Cannot anchor {label}: {traversed}"
                ) from exc
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not _same_inode(before, opened)
                or not _same_inode(opened, after)
            ):
                os.close(child_fd)
                raise SecurePublicationError(
                    f"{label} directory identity changed: {traversed}"
                )
            if created:
                try:
                    _fsync_directory(child_fd, f"new {label} ancestor")
                except Exception:
                    os.close(child_fd)
                    raise
            os.close(current_fd)
            current_fd = child_fd
        _assert_trusted_root_identity(root, root_opened, label)
        opened = os.fstat(current_fd)
        return current_fd, opened, retained_root_fd, root_opened
    except BaseException:
        os.close(current_fd)
        if retained_root_fd is not None:
            os.close(retained_root_fd)
        raise


def _open_directory_anchor(
    path: Path,
    label: str,
    *,
    create: bool,
    trusted_root: Path,
) -> _DirectoryAnchor:
    absolute = _lexical_absolute(path)
    root = _lexical_absolute(trusted_root)
    descriptor, opened, root_descriptor, root_opened = _open_directory_fd_secure(
        absolute, label, create=create, trusted_root=root
    )
    return _DirectoryAnchor(
        path=absolute,
        trusted_root=root,
        label=label,
        descriptor=descriptor,
        opened=opened,
        root_descriptor=root_descriptor,
        root_opened=root_opened,
    )


def _close_anchor(anchor: _DirectoryAnchor) -> None:
    if not anchor.closed:
        try:
            os.close(anchor.descriptor)
        finally:
            try:
                os.close(anchor.root_descriptor)
            finally:
                anchor.closed = True


def _assert_directory_anchor(anchor: _DirectoryAnchor) -> None:
    if anchor.closed:
        raise SecurePublicationError(f"{anchor.label} anchor is closed")
    try:
        held_root = os.fstat(anchor.root_descriptor)
        current = os.fstat(anchor.descriptor)
    except OSError as exc:
        raise SecurePublicationError(
            f"Cannot restat {anchor.label} directory"
        ) from exc
    if not stat.S_ISDIR(held_root.st_mode) or not _same_inode(
        anchor.root_opened, held_root
    ):
        raise SecurePublicationError(
            f"{anchor.label} retained trusted-root identity changed"
        )
    if not stat.S_ISDIR(current.st_mode) or not _same_inode(anchor.opened, current):
        raise SecurePublicationError(f"{anchor.label} directory identity changed")
    root_descriptor, current_root = _open_trusted_root_fd(
        anchor.trusted_root, anchor.label
    )
    try:
        if not _same_inode(held_root, current_root):
            raise SecurePublicationError(
                f"{anchor.label} trusted-root path identity changed"
            )
    finally:
        os.close(root_descriptor)
    descriptor, observed, traversed_root_descriptor, traversed_root = (
        _open_directory_fd_secure(
            anchor.path,
            anchor.label,
            create=False,
            trusted_root=anchor.trusted_root,
        )
    )
    try:
        if (
            not _same_inode(held_root, traversed_root)
            or not _same_inode(
                traversed_root, os.fstat(traversed_root_descriptor)
            )
        ):
            raise SecurePublicationError(
                f"{anchor.label} trusted-root traversal identity changed"
            )
        if not _same_inode(current, observed):
            raise SecurePublicationError(
                f"{anchor.label} directory path identity changed"
            )
    finally:
        os.close(descriptor)
        os.close(traversed_root_descriptor)


def _open_child_directory(
    parent: _DirectoryAnchor, name: str, label: str
) -> tuple[int, os.stat_result]:
    _assert_directory_anchor(parent)
    name = _safe_component(name, label)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise SecurePublicationError(f"Symlink is prohibited for {label}")
        descriptor = os.open(name, _directory_flags(), dir_fd=parent.descriptor)
        opened = os.fstat(descriptor)
        after = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except SecurePublicationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SecurePublicationError(f"Cannot open anchored {label}") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _same_inode(before, opened)
        or not _same_inode(opened, after)
    ):
        os.close(descriptor)
        raise SecurePublicationError(f"Anchored {label} identity differs")
    return descriptor, opened


def _hash_open_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _HASH_BLOCK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _inspect_regular_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    file_validator: FileValidator | None = None,
    expected_stat: os.stat_result | None = None,
    expected_hash: str | None = None,
    _permit_authenticated_ctime_refresh: bool = False,
) -> _ObservedFile:
    name = _safe_component(name, label)
    descriptor: int | None = None
    ctime_drifted = False
    accepted_ctime_ns: int | None = None

    def stable_or_authenticated_ctime(
        left: os.stat_result,
        right: os.stat_result,
    ) -> bool:
        nonlocal accepted_ctime_ns, ctime_drifted
        if _same_stable_file(left, right):
            return True
        metadata_only = (
            _permit_authenticated_ctime_refresh
            and expected_hash is not None
            and left.st_ctime_ns != right.st_ctime_ns
            and _same_stable_file_except_ctime(left, right)
        )
        if metadata_only:
            if accepted_ctime_ns is None:
                accepted_ctime_ns = right.st_ctime_ns
            elif right.st_ctime_ns != accepted_ctime_ns:
                return False
        ctime_drifted = ctime_drifted or metadata_only
        return metadata_only

    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require_regular_links(before, label)
        if expected_stat is not None and not stable_or_authenticated_ctime(
            expected_stat, before
        ):
            raise SecurePublicationError(
                f"Anchored {label} differs from its sealed identity"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
    except SecurePublicationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SecurePublicationError(f"Cannot open anchored {label}") from exc
    try:
        if not stable_or_authenticated_ctime(before, opened):
            raise SecurePublicationError(
                f"Anchored {label} is not the selected regular file"
            )
        digest = _hash_open_descriptor(descriptor)
        if expected_hash is not None and digest != expected_hash:
            raise SecurePublicationError(
                f"Anchored {label} hashes differ from expected; "
                "bundle differs"
            )
        current = os.fstat(descriptor)
        record = FileRecord(name=name, sha256=digest, size_bytes=int(current.st_size))
        if file_validator is not None:
            validation_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            validation_stat = os.fstat(validation_fd)
            if not stable_or_authenticated_ctime(
                current, validation_stat
            ):
                os.close(validation_fd)
                raise SecurePublicationError(
                    f"Anchored {label} changed before validation"
                )
            validation_after: os.stat_result | None = None
            try:
                file_validator(name, validation_fd, record)
            finally:
                try:
                    validation_after = os.fstat(validation_fd)
                except OSError:
                    pass
                if validation_after is not None and _same_inode(
                    validation_stat, validation_after
                ):
                    os.close(validation_fd)
                if validation_after is None or not stable_or_authenticated_ctime(
                    validation_stat, validation_after
                ):
                    raise SecurePublicationError(
                        f"Validator closed, reused, or mutated the borrowed "
                        f"descriptor for {label}"
                    )
        final_descriptor = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stable_or_authenticated_ctime(opened, current)
            or not stable_or_authenticated_ctime(
                current, final_descriptor
            )
            or not stable_or_authenticated_ctime(
                final_descriptor, after
            )
        ):
            raise SecurePublicationError(f"Anchored {label} changed while read")
        # Returning the pre-drift anchor forces the enclosing closed-directory
        # inspection to observe the ctime change and perform one strict second
        # read/hash.  A second drift or any byte change therefore fails closed.
        return _ObservedFile(
            record=record,
            stat=(
                expected_stat
                if ctime_drifted and expected_stat is not None
                else before if ctime_drifted else after
            ),
        )
    finally:
        os.close(descriptor)


def _validate_expected_hashes(
    expected_names: Sequence[str], expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    normalized = {str(name): str(value) for name, value in expected_hashes.items()}
    if set(normalized) != set(expected_names) or len(normalized) != len(expected_names):
        raise SecurePublicationError("Expected hash set differs from closed file set")
    for name, digest in normalized.items():
        _safe_component(name, "expected hash filename")
        if _SHA256_RE.fullmatch(digest) is None:
            raise SecurePublicationError(
                f"Expected SHA-256 is malformed for {name!r}"
            )
    return normalized


def _inspect_closed_directory_fd(
    directory_fd: int,
    expected_names: Sequence[str],
    label: str,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    expected_stats: Mapping[str, os.stat_result] | None = None,
    file_validator: FileValidator | None = None,
    bundle_validator: BundleValidator | None = None,
    _permit_authenticated_ctime_refresh: bool = False,
) -> tuple[BundleSnapshot, dict[str, _ObservedFile]]:
    expected = tuple(sorted(expected_names))
    locked_hashes = (
        None
        if expected_hashes is None
        else _validate_expected_hashes(expected, expected_hashes)
    )
    try:
        names = tuple(sorted(os.listdir(directory_fd)))
    except OSError as exc:
        raise SecurePublicationError(f"Cannot list anchored {label}") from exc
    if names != expected:
        raise SecurePublicationError(
            f"{label} closed file set differs: expected {expected}, actual {names}"
        )
    if expected_stats is not None and set(expected_stats) != set(expected):
        raise SecurePublicationError(
            f"{label} sealed identity set differs from closed file set"
        )
    initial: dict[str, os.stat_result] = {}
    preexisting_ctime_refresh: set[str] = set()
    for name in names:
        selected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require_regular_links(selected, f"{label}/{name}")
        if expected_stats is not None and not _same_stable_file(
            expected_stats[name], selected
        ):
            may_reauthenticate_ctime = (
                _permit_authenticated_ctime_refresh
                and locked_hashes is not None
                and expected_stats[name].st_ctime_ns != selected.st_ctime_ns
                and _same_stable_file_except_ctime(
                    expected_stats[name], selected
                )
            )
            if not may_reauthenticate_ctime:
                raise SecurePublicationError(
                    f"{label}/{name} differs from its sealed identity"
                )
            preexisting_ctime_refresh.add(name)
        initial[name] = selected
    observed = {
        name: _inspect_regular_at(
            directory_fd,
            name,
            f"{label}/{name}",
            file_validator=file_validator,
            expected_stat=initial[name],
            expected_hash=(
                None if locked_hashes is None else locked_hashes[name]
            ),
            _permit_authenticated_ctime_refresh=(
                _permit_authenticated_ctime_refresh
                and name not in preexisting_ctime_refresh
            ),
        )
        for name in names
    }

    ctime_refreshed = set(preexisting_ctime_refresh)

    def reauthenticate(phase: str) -> None:
        try:
            after_names = tuple(sorted(os.listdir(directory_fd)))
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot re-list anchored {label} {phase}"
            ) from exc
        if after_names != expected:
            raise SecurePublicationError(
                f"{label} closed file set changed {phase}"
            )
        for member_name, item in observed.items():
            after = os.stat(
                member_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require_regular_links(after, f"{label}/{member_name}")
            if _same_stable_file(
                initial[member_name], after
            ) and _same_stable_file(item.stat, after):
                continue
            metadata_only_ctime_drift = (
                _permit_authenticated_ctime_refresh
                and member_name not in ctime_refreshed
                and initial[member_name].st_ctime_ns != after.st_ctime_ns
                and item.stat.st_ctime_ns != after.st_ctime_ns
                and _same_stable_file_except_ctime(
                    initial[member_name], after
                )
                and _same_stable_file_except_ctime(item.stat, after)
            )
            if not metadata_only_ctime_drift:
                raise SecurePublicationError(
                    f"{label}/{member_name} changed {phase}"
                )
            refreshed = _inspect_regular_at(
                directory_fd,
                member_name,
                f"{label}/{member_name} ctime refresh",
                expected_stat=after,
            )
            if refreshed.record != item.record:
                raise SecurePublicationError(
                    f"{label}/{member_name} bytes changed {phase}"
                )
            observed[member_name] = refreshed
            initial[member_name] = refreshed.stat
            ctime_refreshed.add(member_name)

    # This is unconditional: a file validator can otherwise add/delete a
    # different member and leave the per-file validation that triggered it
    # looking successful.
    reauthenticate("during file validation")
    if locked_hashes is not None:
        actual = {name: item.record.sha256 for name, item in observed.items()}
        if actual != locked_hashes:
            raise SecurePublicationError(f"{label} hashes differ from expected")
    records = {name: item.record for name, item in observed.items()}
    if bundle_validator is not None:
        bundle_validator(MappingProxyType(records))
        reauthenticate("during bundle validation")
    snapshot = BundleSnapshot(tuple(records[name] for name in expected))
    return snapshot, observed


def _read_regular_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    expected_nlink: int = 1,
    expected_stat: os.stat_result | None = None,
    fsync: bool = False,
) -> bytes:
    name = _safe_component(name, label)
    descriptor: int | None = None
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require_regular_links(before, label, expected_nlink=expected_nlink)
        if expected_stat is not None and not _same_stable_regular(
            expected_stat, before, expected_nlink=expected_nlink
        ):
            raise SecurePublicationError(f"Anchored {label} identity differs")
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
    except SecurePublicationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SecurePublicationError(f"Cannot read anchored {label}") from exc
    try:
        if not _same_stable_regular(
            before, opened, expected_nlink=expected_nlink
        ):
            raise SecurePublicationError(f"Anchored {label} identity differs")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        if fsync:
            _fsync_file(descriptor, label)
        current = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not _same_stable_regular(
            opened, current, expected_nlink=expected_nlink
        ) or not _same_stable_regular(
            current, after, expected_nlink=expected_nlink
        ):
            raise SecurePublicationError(f"Anchored {label} changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SecurePublicationError("Short write while publishing payload")
        remaining = remaining[written:]


def _write_payload_at(
    directory_fd: int, name: str, payload: bytes, label: str
) -> os.stat_result:
    name = _safe_component(name, label)
    descriptor: int | None = None
    opened: os.stat_result | None = None
    current: os.stat_result | None = None
    failure: BaseException | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        _require_regular_links(opened, label)
        _write_all(descriptor, payload)
        _fsync_file(descriptor, label)
        current = os.fstat(descriptor)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not _same_regular_inode(opened, current)
            or not _same_regular_inode(current, after)
            or current.st_size != len(payload)
            or after.st_size != len(payload)
        ):
            raise SecurePublicationError(f"Anchored {label} identity changed")
    except BaseException as exc:
        failure = exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if failure is not None:
        if opened is not None:
            try:
                selected = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                selected = None
            except OSError as cleanup_exc:
                raise SecurePublicationError(
                    f"Cannot authenticate failed {label} cleanup"
                ) from cleanup_exc
            if selected is not None:
                if not _same_inode(opened, selected):
                    raise SecurePublicationError(
                        f"Failed {label} cleanup found an unowned entry; preserved"
                    ) from failure
                try:
                    _retire_owned_entry_at(
                        directory_fd,
                        name,
                        selected,
                        f"failed {label}",
                    )
                except (OSError, SecurePublicationError) as cleanup_exc:
                    raise SecurePublicationError(
                        f"Cannot safely retire failed {label}"
                    ) from cleanup_exc
        if isinstance(failure, SecurePublicationError):
            raise failure
        raise SecurePublicationError(f"Cannot create anchored {label}") from failure
    assert current is not None
    return current


def _fsync_authenticated_members(
    directory_fd: int,
    expected_stats: Mapping[str, os.stat_result],
    label: str,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    _permit_authenticated_ctime_refresh: bool = False,
) -> dict[str, os.stat_result]:
    """Fsync the exact sealed members and re-authenticate their path entries."""

    expected_names = tuple(sorted(expected_stats))
    locked_hashes = (
        _validate_expected_hashes(expected_names, expected_hashes)
        if expected_hashes is not None
        else None
    )
    if _permit_authenticated_ctime_refresh and locked_hashes is None:
        raise SecurePublicationError(
            f"{label} ctime refresh requires locked member hashes"
        )
    if tuple(sorted(os.listdir(directory_fd))) != expected_names:
        raise SecurePublicationError(
            f"{label} closed file set changed before final member fsync"
        )
    refreshed_stats: dict[str, os.stat_result] = {}
    for name in expected_names:
        descriptor: int | None = None
        try:
            if _permit_authenticated_ctime_refresh:
                assert locked_hashes is not None
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_fd,
                )
                authenticated = _rehash_retained_regular_stable(
                    directory_fd=directory_fd,
                    name=name,
                    descriptor=descriptor,
                    baseline=expected_stats[name],
                    expected_sha256=locked_hashes[name],
                    label=f"{label}/{name} before final member fsync",
                )
                _fsync_file(
                    descriptor,
                    f"final authenticated {label} member {name}",
                )
                try:
                    after_fsync = _rehash_retained_regular_stable(
                        directory_fd=directory_fd,
                        name=name,
                        descriptor=descriptor,
                        baseline=authenticated,
                        expected_sha256=locked_hashes[name],
                        label=f"{label}/{name} after final member fsync",
                    )
                except SecurePublicationError as exc:
                    if "SHA-256 changed" in str(exc):
                        raise SecurePublicationError(
                            f"{label}/{name} bytes changed during "
                            "final member fsync"
                        ) from exc
                    raise
                if not _same_stable_file(authenticated, after_fsync):
                    _fsync_file(
                        descriptor,
                        f"durability after authenticated {label} member "
                        f"{name} ctime refresh",
                    )
                    after_fsync = _rehash_retained_regular_stable(
                        directory_fd=directory_fd,
                        name=name,
                        descriptor=descriptor,
                        baseline=after_fsync,
                        expected_sha256=locked_hashes[name],
                        label=(
                            f"{label}/{name} after ctime-refresh "
                            "durability fsync"
                        ),
                    )
                refreshed_stats[name] = after_fsync
                continue
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            refreshed_before_fsync = False
            if not _same_stable_file(expected_stats[name], before):
                may_reauthenticate_ctime = (
                    _permit_authenticated_ctime_refresh
                    and expected_stats[name].st_ctime_ns != before.st_ctime_ns
                    and _same_stable_file_except_ctime(
                        expected_stats[name], before
                    )
                )
                if not may_reauthenticate_ctime:
                    raise SecurePublicationError(
                        f"{label}/{name} differs from its sealed identity before fsync"
                    )
                refreshed = _inspect_regular_at(
                    directory_fd,
                    name,
                    f"{label}/{name} before final member fsync ctime refresh",
                    expected_stat=before,
                )
                assert locked_hashes is not None
                if refreshed.record.sha256 != locked_hashes[name]:
                    raise SecurePublicationError(
                        f"{label}/{name} bytes changed before final member fsync"
                    )
                before = refreshed.stat
                refreshed_before_fsync = True
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
            if not _same_stable_file(before, opened):
                raise SecurePublicationError(
                    f"{label}/{name} changed before final member fsync"
                )
            _fsync_file(descriptor, f"final authenticated {label} member {name}")
            after_descriptor = os.fstat(descriptor)
            after_path = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            stable_after_fsync = _same_stable_file(
                opened, after_descriptor
            ) and _same_stable_file(after_descriptor, after_path)
            if not stable_after_fsync:
                may_reauthenticate_ctime = (
                    _permit_authenticated_ctime_refresh
                    and not refreshed_before_fsync
                    and opened.st_ctime_ns != after_descriptor.st_ctime_ns
                    and _same_stable_file_except_ctime(
                        opened, after_descriptor
                    )
                    and _same_stable_file(after_descriptor, after_path)
                )
                if not may_reauthenticate_ctime:
                    raise SecurePublicationError(
                        f"{label}/{name} changed during final member fsync"
                    )
                assert locked_hashes is not None
                if _hash_open_descriptor(descriptor) != locked_hashes[name]:
                    raise SecurePublicationError(
                        f"{label}/{name} bytes changed during final member fsync"
                    )
                final_descriptor = os.fstat(descriptor)
                final_path = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not _same_stable_file(after_descriptor, final_descriptor)
                    or not _same_stable_file(final_descriptor, final_path)
                ):
                    raise SecurePublicationError(
                        f"{label}/{name} changed during final fsync reauthentication"
                    )
                _fsync_file(
                    descriptor,
                    f"durability after authenticated {label} member {name} "
                    "ctime refresh",
                )
                durable_descriptor = os.fstat(descriptor)
                durable_path = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not _same_stable_file(final_descriptor, durable_descriptor)
                    or not _same_stable_file(durable_descriptor, durable_path)
                ):
                    raise SecurePublicationError(
                        f"{label}/{name} changed during ctime-refresh durability fsync"
                    )
                after_path = durable_path
            refreshed_stats[name] = after_path
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot fsync authenticated {label}/{name}"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
    if tuple(sorted(os.listdir(directory_fd))) != expected_names:
        raise SecurePublicationError(
            f"{label} closed file set changed during final member fsync"
        )
    return refreshed_stats


def _read_held_regular_descriptor(
    descriptor: int,
    expected: os.stat_result,
    label: str,
) -> bytes:
    """Read one retained regular-file descriptor without changing identity."""

    try:
        before = os.fstat(descriptor)
        if not _same_stable_file(expected, before):
            raise SecurePublicationError(f"Held {label} identity changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except SecurePublicationError:
        raise
    except OSError as exc:
        raise SecurePublicationError(f"Cannot read held {label}") from exc
    if not _same_stable_file(before, after):
        raise SecurePublicationError(f"Held {label} changed while read")
    return b"".join(chunks)


def _retained_stat_matches(
    baseline: os.stat_result,
    observed: os.stat_result,
) -> bool:
    """Match one retained inode while allowing only ctime to differ."""

    return _same_stable_file(baseline, observed) or (
        baseline.st_ctime_ns != observed.st_ctime_ns
        and _same_stable_file_except_ctime(baseline, observed)
    )


def _read_retained_payload_stable(
    *,
    directory_fd: int,
    name: str,
    descriptor: int,
    baseline: os.stat_result,
    label: str,
    expected_payload: bytes | None = None,
    expected_sha256: str | None = None,
    attempts: int = 2,
) -> tuple[bytes, os.stat_result]:
    """Read a retained file through bounded ctime-only metadata races.

    A pass is accepted only when the held descriptor and relative path retain
    the original inode/mode/link/owner/size/mtime identity, the payload matches
    any locked bytes/hash, and those stats remain stable for the entire final
    read.  A ctime-only race triggers another complete read.
    """

    name = _safe_component(name, label)
    prior_payload: bytes | None = None
    for _attempt in range(attempts):
        try:
            before_descriptor = os.fstat(descriptor)
            before_path = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require_regular_links(before_descriptor, label)
            _require_regular_links(before_path, label)
            if (
                not _retained_stat_matches(baseline, before_descriptor)
                or not _retained_stat_matches(baseline, before_path)
                or not _retained_stat_matches(
                    before_descriptor, before_path
                )
            ):
                raise SecurePublicationError(
                    f"Anchored {label} path identity changed"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1 << 20):
                chunks.append(chunk)
            payload = b"".join(chunks)
            after_descriptor = os.fstat(descriptor)
            after_path = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except SecurePublicationError:
            raise
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot read retained {label}"
            ) from exc
        if (
            not _retained_stat_matches(baseline, after_descriptor)
            or not _retained_stat_matches(baseline, after_path)
            or not _retained_stat_matches(
                after_descriptor, after_path
            )
        ):
            raise SecurePublicationError(
                f"Anchored {label} path identity changed during read"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if expected_payload is not None and payload != expected_payload:
            raise SecurePublicationError(
                f"Anchored {label} payload changed"
            )
        if expected_sha256 is not None and digest != expected_sha256:
            raise SecurePublicationError(
                f"Anchored {label} SHA-256 changed"
            )
        if prior_payload is not None and payload != prior_payload:
            raise SecurePublicationError(
                f"Anchored {label} payload changed across ctime refresh"
            )
        stable_pass = (
            _same_stable_file(before_descriptor, before_path)
            and _same_stable_file(
                before_descriptor, after_descriptor
            )
            and _same_stable_file(after_descriptor, after_path)
        )
        if stable_pass:
            return payload, after_path
        prior_payload = payload
    raise SecurePublicationError(
        f"Anchored {label} ctime did not stabilize during authentication"
    )


def _rehash_retained_regular_stable(
    *,
    directory_fd: int,
    name: str,
    descriptor: int,
    baseline: os.stat_result,
    expected_sha256: str,
    label: str,
    attempts: int = 2,
) -> os.stat_result:
    """Rehash a retained inode until one ctime-stable pass succeeds."""

    name = _safe_component(name, label)
    for _attempt in range(attempts):
        try:
            before_descriptor = os.fstat(descriptor)
            before_path = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            _require_regular_links(before_descriptor, label)
            _require_regular_links(before_path, label)
            if (
                not _retained_stat_matches(baseline, before_descriptor)
                or not _retained_stat_matches(baseline, before_path)
                or not _retained_stat_matches(
                    before_descriptor, before_path
                )
            ):
                raise SecurePublicationError(
                    f"Anchored {label} path identity changed"
                )
            digest = _hash_open_descriptor(descriptor)
            after_descriptor = os.fstat(descriptor)
            after_path = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except SecurePublicationError:
            raise
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot rehash retained {label}"
            ) from exc
        if digest != expected_sha256:
            raise SecurePublicationError(
                f"Anchored {label} SHA-256 changed"
            )
        if (
            not _retained_stat_matches(baseline, after_descriptor)
            or not _retained_stat_matches(baseline, after_path)
            or not _retained_stat_matches(
                after_descriptor, after_path
            )
        ):
            raise SecurePublicationError(
                f"Anchored {label} path identity changed during rehash"
            )
        if (
            _same_stable_file(before_descriptor, before_path)
            and _same_stable_file(
                before_descriptor, after_descriptor
            )
            and _same_stable_file(after_descriptor, after_path)
        ):
            return after_path
    raise SecurePublicationError(
        f"Anchored {label} ctime did not stabilize during rehash"
    )


def _validate_guarded_payload(
    payload: bytes,
    label: str,
    *,
    expected_payload: bytes | None,
    expected_sha256: str | None,
    payload_validator: PayloadValidator | None,
) -> str:
    """Validate one captured pointer payload and return its digest."""

    digest = hashlib.sha256(payload).hexdigest()
    if expected_payload is not None and payload != expected_payload:
        raise SecurePublicationError(f"{label} payload differs from expected")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SecurePublicationError(f"{label} SHA-256 differs from expected")
    if payload_validator is not None:
        payload_validator(payload)
    return digest


class AuthenticatedPointerGuard:
    """Retain and revalidate one existing regular authority pointer.

    Construct a guard with :meth:`open`, inspect :attr:`payload`, and pass
    :meth:`gate` to publication pre/post gates.  The parent directory and the
    exact regular-file inode remain open until :meth:`close` or context exit.
    """

    def __init__(
        self,
        *,
        path: Path,
        label: str,
        parent: _DirectoryAnchor,
        descriptor: int,
        opened: os.stat_result,
        payload: bytes,
        payload_validator: PayloadValidator | None,
    ) -> None:
        self.path = path
        self.label = label
        self._parent = parent
        self._descriptor = descriptor
        self._opened = opened
        self._payload = payload
        self._payload_sha256 = hashlib.sha256(payload).hexdigest()
        self._record = FileRecord(
            name=path.name,
            sha256=self._payload_sha256,
            size_bytes=len(payload),
        )
        self._payload_validator = payload_validator
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        trusted_root: Path,
        path: Path,
        expected_payload: bytes | None = None,
        expected_sha256: str | None = None,
        payload_validator: PayloadValidator | None = None,
        label: str = "authority pointer",
    ) -> "AuthenticatedPointerGuard":
        """Open and authenticate an existing pointer beneath ``trusted_root``."""

        if expected_payload is not None and not isinstance(expected_payload, bytes):
            raise SecurePublicationError("Expected pointer payload must be bytes")
        if expected_sha256 is not None and _SHA256_RE.fullmatch(
            str(expected_sha256)
        ) is None:
            raise SecurePublicationError("Expected pointer SHA-256 is malformed")
        locked_sha256 = (
            str(expected_sha256) if expected_sha256 is not None else None
        )
        root = _lexical_absolute(trusted_root)
        pointer = _lexical_absolute(path)
        relative = _relative_beneath(pointer, root, label)
        if not relative.parts:
            raise SecurePublicationError("Pointer cannot equal trusted root")
        name = _safe_component(pointer.name, label)
        parent = _open_directory_anchor(
            pointer.parent,
            f"{label} parent",
            create=False,
            trusted_root=root,
        )
        descriptor: int | None = None
        try:
            try:
                before = os.stat(
                    name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                _require_regular_links(before, label)
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent.descriptor,
                )
                opened = os.fstat(descriptor)
                if not _retained_stat_matches(before, opened):
                    raise SecurePublicationError(
                        f"Anchored {label} is not the selected regular file"
                    )
                payload, retained = _read_retained_payload_stable(
                    directory_fd=parent.descriptor,
                    name=name,
                    descriptor=descriptor,
                    baseline=before,
                    label=label,
                    expected_payload=expected_payload,
                    expected_sha256=locked_sha256,
                )
                _validate_guarded_payload(
                    payload,
                    label,
                    expected_payload=expected_payload,
                    expected_sha256=locked_sha256,
                    payload_validator=payload_validator,
                )
                confirmed_payload, after = _read_retained_payload_stable(
                    directory_fd=parent.descriptor,
                    name=name,
                    descriptor=descriptor,
                    baseline=retained,
                    label=label,
                    expected_payload=payload,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )
            except SecurePublicationError:
                raise
            except OSError as exc:
                raise SecurePublicationError(
                    f"Cannot open existing {label}"
                ) from exc
            if confirmed_payload != payload:
                raise SecurePublicationError(
                    f"Existing {label} changed while opening guard"
                )
            _assert_directory_anchor(parent)
            return cls(
                path=pointer,
                label=label,
                parent=parent,
                descriptor=descriptor,
                opened=after,
                payload=payload,
                payload_validator=payload_validator,
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            _close_anchor(parent)
            raise

    @property
    def payload(self) -> bytes:
        """Return the immutable payload captured when the guard opened."""

        return self._payload

    @property
    def payload_sha256(self) -> str:
        """Return the SHA-256 digest of :attr:`payload`."""

        return self._payload_sha256

    @property
    def record(self) -> FileRecord:
        """Return the captured pointer's immutable content record."""

        return self._record

    @property
    def closed(self) -> bool:
        return self._closed

    def _assert_open(self) -> None:
        if self._closed:
            raise SecurePublicationError(f"{self.label} guard is closed")

    def revalidate(self) -> bytes:
        """Reauthenticate the retained payload, inode, basename, and parent."""

        self._assert_open()
        name = self.path.name
        try:
            _assert_directory_anchor(self._parent)
            payload, retained = _read_retained_payload_stable(
                directory_fd=self._parent.descriptor,
                name=name,
                descriptor=self._descriptor,
                baseline=self._opened,
                label=self.label,
                expected_payload=self._payload,
                expected_sha256=self._payload_sha256,
            )
            _validate_guarded_payload(
                payload,
                self.label,
                expected_payload=self._payload,
                expected_sha256=self._payload_sha256,
                payload_validator=self._payload_validator,
            )
            try:
                confirmed_payload, after_path = (
                    _read_retained_payload_stable(
                        directory_fd=self._parent.descriptor,
                        name=name,
                        descriptor=self._descriptor,
                        baseline=retained,
                        label=self.label,
                        expected_payload=self._payload,
                        expected_sha256=self._payload_sha256,
                    )
                )
            except SecurePublicationError as exc:
                if "path identity changed" in str(exc):
                    raise SecurePublicationError(
                        f"Anchored {self.label} path identity changed "
                        "during revalidation"
                    ) from exc
                raise
        except SecurePublicationError:
            raise
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot revalidate {self.label}"
            ) from exc
        if payload != self._payload or confirmed_payload != self._payload:
            raise SecurePublicationError(
                f"Anchored {self.label} payload changed during revalidation"
            )
        self._opened = after_path
        _assert_directory_anchor(self._parent)
        return payload

    def gate(self) -> None:
        """Gate callback that rejects any change since :meth:`open`."""

        self.revalidate()

    def close(self) -> None:
        """Close the retained file and parent-directory descriptors."""

        if self._closed:
            return
        failure: BaseException | None = None
        try:
            os.close(self._descriptor)
        except OSError as exc:
            failure = SecurePublicationError(
                f"Cannot close {self.label} guard descriptor"
            )
            failure.__cause__ = exc
        try:
            _close_anchor(self._parent)
        except OSError as exc:
            if failure is None:
                failure = SecurePublicationError(
                    f"Cannot close {self.label} guard parent"
                )
                failure.__cause__ = exc
        self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> "AuthenticatedPointerGuard":
        self._assert_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            self.close()
        except SecurePublicationError as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"Authenticated pointer guard cleanup: {cleanup_error}")
        return False


class AuthenticatedBundleGuard:
    """Retain and revalidate one existing immutable flat closed bundle."""

    def __init__(
        self,
        *,
        path: Path,
        label: str,
        expected_files: tuple[str, ...],
        expected_hashes: Mapping[str, str],
        file_validator: FileValidator | None,
        bundle_validator: BundleValidator | None,
        parent: _DirectoryAnchor,
        descriptor: int,
        opened: os.stat_result,
        snapshot: BundleSnapshot,
        member_stats: Mapping[str, os.stat_result],
        member_descriptors: Mapping[str, int],
        permit_authenticated_directory_ctime_refresh: bool,
    ) -> None:
        self.path = path
        self.label = label
        self.expected_files = expected_files
        self.expected_hashes = MappingProxyType(dict(expected_hashes))
        self._file_validator = file_validator
        self._bundle_validator = bundle_validator
        self._parent = parent
        self._descriptor = descriptor
        self._opened = opened
        self._snapshot = snapshot
        self._member_stats = dict(member_stats)
        self._member_descriptors = dict(member_descriptors)
        self._permit_authenticated_directory_ctime_refresh = (
            permit_authenticated_directory_ctime_refresh
        )
        self._directory_ctime_refreshed = False
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        trusted_root: Path,
        path: Path,
        expected_files: Sequence[str],
        expected_hashes: Mapping[str, str],
        file_validator: FileValidator | None = None,
        bundle_validator: BundleValidator | None = None,
        label: str = "immutable bundle",
        _permit_authenticated_directory_ctime_refresh: bool = False,
    ) -> "AuthenticatedBundleGuard":
        """Open and authenticate an existing immutable closed bundle."""

        root = _lexical_absolute(trusted_root)
        bundle = _lexical_absolute(path)
        relative = _relative_beneath(bundle, root, label)
        if not relative.parts:
            raise SecurePublicationError(
                "Immutable bundle cannot equal trusted root"
            )
        name = _safe_component(bundle.name, f"{label} destination")
        names = tuple(
            sorted(
                _safe_component(member, "closed file")
                for member in expected_files
            )
        )
        if not names or len(names) != len(set(names)):
            raise SecurePublicationError(
                "Closed file set must be nonempty and contain unique safe basenames"
            )
        locked_hashes = _validate_expected_hashes(names, expected_hashes)
        parent = _open_directory_anchor(
            bundle.parent,
            f"{label} parent",
            create=False,
            trusted_root=root,
        )
        descriptor: int | None = None
        member_descriptors: dict[str, int] = {}
        retained_stats: dict[str, os.stat_result] = {}
        try:
            descriptor, opened = _open_child_directory(parent, name, label)
            snapshot, observed = _inspect_closed_directory_fd(
                descriptor,
                names,
                label,
                expected_hashes=locked_hashes,
                file_validator=file_validator,
                bundle_validator=bundle_validator,
                _permit_authenticated_ctime_refresh=True,
            )
            for member in names:
                member_descriptor = os.open(
                    member,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                member_descriptors[member] = member_descriptor
                retained_stats[member] = (
                    _rehash_retained_regular_stable(
                        directory_fd=descriptor,
                        name=member,
                        descriptor=member_descriptor,
                        baseline=observed[member].stat,
                        expected_sha256=locked_hashes[member],
                        label=f"{label}/{member}",
                    )
                )
            for member in names:
                retained_stats[member] = (
                    _rehash_retained_regular_stable(
                        directory_fd=descriptor,
                        name=member,
                        descriptor=member_descriptors[member],
                        baseline=retained_stats[member],
                        expected_sha256=locked_hashes[member],
                        label=f"{label}/{member} after inode retention",
                    )
                )
            if tuple(sorted(os.listdir(descriptor))) != names:
                raise SecurePublicationError(
                    f"Anchored {label} closed file set changed while opening guard"
                )
            current = os.fstat(descriptor)
            relative_stat = os.stat(
                name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            if not _same_stable_directory(current, relative_stat):
                raise SecurePublicationError(
                    f"Anchored {label} directory changed while opening guard"
                )
            if not _same_stable_directory(opened, current):
                if not (
                    _permit_authenticated_directory_ctime_refresh
                    and _same_stable_directory_except_ctime(
                        opened, current
                    )
                ):
                    raise SecurePublicationError(
                        f"Anchored {label} directory changed while opening guard"
                    )
                opened = current
            _assert_directory_anchor(parent)
            return cls(
                path=bundle,
                label=label,
                expected_files=names,
                expected_hashes=locked_hashes,
                file_validator=file_validator,
                bundle_validator=bundle_validator,
                parent=parent,
                descriptor=descriptor,
                opened=opened,
                snapshot=snapshot,
                member_stats=retained_stats,
                member_descriptors=member_descriptors,
                permit_authenticated_directory_ctime_refresh=(
                    _permit_authenticated_directory_ctime_refresh
                ),
            )
        except BaseException as exc:
            for member_descriptor in member_descriptors.values():
                os.close(member_descriptor)
            if descriptor is not None:
                os.close(descriptor)
            _close_anchor(parent)
            if isinstance(exc, OSError):
                raise SecurePublicationError(
                    f"Cannot open existing {label}"
                ) from exc
            raise

    @property
    def snapshot(self) -> BundleSnapshot:
        """Return the authenticated snapshot captured when the guard opened."""

        return self._snapshot

    @property
    def closed(self) -> bool:
        return self._closed

    def _assert_open(self) -> None:
        if self._closed:
            raise SecurePublicationError(f"{self.label} guard is closed")

    def _observe_bundle_identity(self) -> os.stat_result:
        _assert_directory_anchor(self._parent)
        descriptor_stat = os.fstat(self._descriptor)
        path_stat = os.stat(
            self.path.name,
            dir_fd=self._parent.descriptor,
            follow_symlinks=False,
        )
        if not _same_stable_directory(descriptor_stat, path_stat):
            raise SecurePublicationError(
                f"Anchored {self.label} directory identity changed"
            )
        if tuple(sorted(os.listdir(self._descriptor))) != self.expected_files:
            raise SecurePublicationError(
                f"Anchored {self.label} closed file set changed"
            )
        return descriptor_stat

    def _assert_bundle_identity(self) -> None:
        observed = self._observe_bundle_identity()
        if not _same_stable_directory(self._opened, observed):
            raise SecurePublicationError(
                f"Anchored {self.label} directory identity changed"
            )

    def _may_refresh_directory_ctime(self, observed: os.stat_result) -> bool:
        return (
            self._permit_authenticated_directory_ctime_refresh
            and not self._directory_ctime_refreshed
            and self._opened.st_ctime_ns != observed.st_ctime_ns
            and _same_stable_directory_except_ctime(self._opened, observed)
        )

    def _revalidate_retained_members(
        self,
        members: Sequence[str] | None = None,
        *,
        baselines: Mapping[str, os.stat_result] | None = None,
    ) -> dict[str, os.stat_result]:
        selected_members = self.expected_files if members is None else members
        candidate = dict(
            self._member_stats if baselines is None else baselines
        )
        for member in selected_members:
            candidate[member] = _rehash_retained_regular_stable(
                directory_fd=self._descriptor,
                name=member,
                descriptor=self._member_descriptors[member],
                baseline=candidate[member],
                expected_sha256=self.expected_hashes[member],
                label=f"held {self.label}/{member}",
            )
        return candidate

    def revalidate(self) -> BundleSnapshot:
        """Reauthenticate the held directory, closed set, hashes, and members."""

        self._assert_open()
        try:
            before_directory = self._observe_bundle_identity()
            refresh_before = not _same_stable_directory(
                self._opened, before_directory
            )
            if refresh_before and not self._may_refresh_directory_ctime(
                before_directory
            ):
                raise SecurePublicationError(
                    f"Anchored {self.label} directory identity changed"
                )
            candidate_stats = self._revalidate_retained_members()
            snapshot, observed = _inspect_closed_directory_fd(
                self._descriptor,
                self.expected_files,
                self.label,
                expected_hashes=self.expected_hashes,
                expected_stats=candidate_stats,
                file_validator=self._file_validator,
                bundle_validator=self._bundle_validator,
                _permit_authenticated_ctime_refresh=True,
            )
            candidate_stats = self._revalidate_retained_members(
                baselines={
                    member: item.stat
                    for member, item in observed.items()
                }
            )
            after_directory = self._observe_bundle_identity()
            changed_during_revalidation = not _same_stable_directory(
                before_directory, after_directory
            )
            if changed_during_revalidation:
                if refresh_before or not self._may_refresh_directory_ctime(
                    after_directory
                ) or not _same_stable_directory_except_ctime(
                    before_directory, after_directory
                ):
                    raise SecurePublicationError(
                        f"Anchored {self.label} directory identity changed "
                        "during revalidation"
                    )
            refresh_after = refresh_before or changed_during_revalidation
        except SecurePublicationError:
            raise
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot revalidate {self.label}"
            ) from exc
        if snapshot != self._snapshot:
            raise SecurePublicationError(
                f"Anchored {self.label} snapshot changed during revalidation"
            )
        self._member_stats = candidate_stats
        if refresh_after:
            self._opened = after_directory
            self._directory_ctime_refreshed = True
        return snapshot

    @contextlib.contextmanager
    def open_binary(self, name: str) -> Iterator[BinaryIO]:
        """Yield a read-only stream for one retained member inode.

        The yielded stream owns only a duplicate descriptor.  The guard keeps
        its retained member descriptor and revalidates the complete bundle
        before and after access.
        """

        self._assert_open()
        member = _safe_component(name, "guarded bundle member")
        if member not in self._member_descriptors:
            raise SecurePublicationError(
                f"Unknown guarded bundle member: {member!r}"
        )
        try:
            self._assert_bundle_identity()
            pre_access_stats = self._revalidate_retained_members(
                (member,)
            )
            self._assert_bundle_identity()
            self._member_stats = pre_access_stats
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot authenticate guarded {self.label}/{member}"
            ) from exc
        descriptor: int | None = None
        stream: io.FileIO | None = None
        failure: BaseException | None = None
        try:
            try:
                descriptor = os.open(
                    member,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=self._descriptor,
                )
            except OSError as exc:
                raise SecurePublicationError(
                    f"Cannot open guarded {self.label}/{member}"
                ) from exc
            access_stat = _rehash_retained_regular_stable(
                directory_fd=self._descriptor,
                name=member,
                descriptor=descriptor,
                baseline=self._member_stats[member],
                expected_sha256=self.expected_hashes[member],
                label=f"guarded {self.label}/{member} before access",
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            stream = os.fdopen(os.dup(descriptor), "rb", buffering=0)
            try:
                yield stream
            except BaseException as exc:
                failure = exc
                raise
        finally:
            if stream is not None and not stream.closed:
                stream.close()
            validation_failure: BaseException | None = None
            post_access_stat: os.stat_result | None = None
            if descriptor is not None:
                try:
                    post_access_stat = (
                        _rehash_retained_regular_stable(
                            directory_fd=self._descriptor,
                            name=member,
                            descriptor=descriptor,
                            baseline=access_stat,
                            expected_sha256=self.expected_hashes[
                                member
                            ],
                            label=(
                                f"guarded {self.label}/{member} "
                                "after access"
                            ),
                        )
                    )
                except BaseException as exc:
                    validation_failure = exc
                finally:
                    os.close(descriptor)
            if validation_failure is None:
                try:
                    assert post_access_stat is not None
                    baselines = {
                        **self._member_stats,
                        member: post_access_stat,
                    }
                    post_access_stats = (
                        self._revalidate_retained_members(
                            (member,),
                            baselines=baselines,
                        )
                    )
                    self._assert_bundle_identity()
                    self._member_stats = post_access_stats
                except BaseException as exc:
                    validation_failure = exc
            if validation_failure is not None:
                if failure is not None and hasattr(failure, "add_note"):
                    failure.add_note(
                        f"Guarded bundle access revalidation: {validation_failure}"
                    )
                elif failure is None:
                    raise validation_failure

    def gate(self) -> None:
        """Gate callback that rejects any change since :meth:`open`."""

        self.revalidate()

    def close(self) -> None:
        """Close the retained bundle and parent-directory descriptors."""

        if self._closed:
            return
        failure: BaseException | None = None
        for member, member_descriptor in self._member_descriptors.items():
            try:
                os.close(member_descriptor)
            except OSError as exc:
                if failure is None:
                    failure = SecurePublicationError(
                        f"Cannot close {self.label}/{member} retained descriptor"
                    )
                    failure.__cause__ = exc
        try:
            os.close(self._descriptor)
        except OSError as exc:
            failure = SecurePublicationError(
                f"Cannot close {self.label} guard descriptor"
            )
            failure.__cause__ = exc
        try:
            _close_anchor(self._parent)
        except OSError as exc:
            if failure is None:
                failure = SecurePublicationError(
                    f"Cannot close {self.label} guard parent"
                )
                failure.__cause__ = exc
        self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> "AuthenticatedBundleGuard":
        self._assert_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            self.close()
        except SecurePublicationError as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"Authenticated bundle guard cleanup: {cleanup_error}")
        return False


class SecureStagingDirectory:
    """One anchored staging directory for a flat immutable closed bundle.

    Construct instances with :meth:`create` and use them as context managers.
    A normal context exit discards a stage that was not published.  Once every
    member has been written, call :meth:`snapshot` to seal the stage and obtain
    its physical hashes, then call :meth:`publish` with those expected hashes.
    """

    def __init__(
        self,
        *,
        destination: Path,
        expected_files: tuple[str, ...],
        parent: _DirectoryAnchor,
        stage_name: str,
        descriptor: int,
        opened: os.stat_result,
    ) -> None:
        self.destination = destination
        self.expected_files = expected_files
        self._parent = parent
        self._stage_name = stage_name
        self._descriptor = descriptor
        self._opened = opened
        self._created: dict[str, os.stat_result] = {}
        self._sealed_stats: dict[str, os.stat_result] | None = None
        self._sealed = False
        self._published = False
        self._discarded = False
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        trusted_root: Path,
        destination: Path,
        expected_files: Sequence[str],
        label: str = "immutable publication",
    ) -> "SecureStagingDirectory":
        """Create an exclusive sibling stage beneath ``trusted_root``."""

        root = _lexical_absolute(trusted_root)
        final = _lexical_absolute(destination)
        relative = _relative_beneath(final, root, label)
        if not relative.parts:
            raise SecurePublicationError("Immutable destination cannot be trusted root")
        _safe_component(final.name, f"{label} destination")
        names = tuple(sorted(_safe_component(name, "closed file") for name in expected_files))
        if not names or len(names) != len(set(names)):
            raise SecurePublicationError(
                "Closed file set must be nonempty and contain unique safe basenames"
            )
        parent = _open_directory_anchor(
            final.parent,
            f"{label} parent",
            create=True,
            trusted_root=root,
        )
        descriptor: int | None = None
        stage_name: str | None = None
        stage_stat: os.stat_result | None = None
        try:
            try:
                existing = os.stat(
                    final.name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and (
                stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode)
            ):
                raise SecurePublicationError(
                    f"Existing {label} destination is not a real directory"
                )
            for _attempt in range(128):
                candidate = (
                    f".{_safe_component(final.name, 'stage prefix')}."
                    f"staging-{secrets.token_hex(8)}"
                )
                try:
                    os.mkdir(candidate, 0o700, dir_fd=parent.descriptor)
                    stage_name = candidate
                    stage_stat = os.stat(
                        candidate,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                    break
                except FileExistsError:
                    continue
            else:
                raise SecurePublicationError("Cannot allocate exclusive staging directory")
            _fsync_directory(parent.descriptor, f"{label} parent after stage mkdir")
            descriptor, opened = _open_child_directory(
                parent, stage_name, f"{label} staging directory"
            )
            if stage_stat is None or not _same_stable_directory(
                stage_stat, opened
            ):
                raise SecurePublicationError(
                    f"{label} staging directory changed between mkdir and open"
                )
            return cls(
                destination=final,
                expected_files=names,
                parent=parent,
                stage_name=stage_name,
                descriptor=descriptor,
                opened=opened,
            )
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if stage_name is not None and stage_stat is not None:
                try:
                    observed = os.stat(
                        stage_name,
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                    if _same_inode(stage_stat, observed):
                        _retire_owned_entry_at(
                            parent.descriptor,
                            stage_name,
                            observed,
                            f"failed {label} staging directory",
                        )
                except (OSError, SecurePublicationError):
                    pass
            _close_anchor(parent)
            raise

    def __enter__(self) -> "SecureStagingDirectory":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            self.close()
        except SecurePublicationError as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"Secure staging cleanup: {cleanup_error}")
        return False

    def _assert_open(self) -> None:
        if self._closed:
            raise SecurePublicationError("Staging directory is closed")
        if self._published:
            raise SecurePublicationError("Staging directory is already published")
        _assert_directory_anchor(self._parent)
        current = os.fstat(self._descriptor)
        if not stat.S_ISDIR(current.st_mode) or not _same_inode(self._opened, current):
            raise SecurePublicationError("Staging directory identity changed")
        relative = os.stat(
            self._stage_name,
            dir_fd=self._parent.descriptor,
            follow_symlinks=False,
        )
        if not _same_inode(current, relative):
            raise SecurePublicationError("Staging path identity changed")

    @contextlib.contextmanager
    def open_exclusive_binary(self, name: str) -> Iterator[BinaryIO]:
        """Create one member and yield a binary stream suitable for PyArrow.

        The yielded stream writes through a duplicate descriptor.  A consumer
        may close it (as PyArrow commonly does); the original anchored
        descriptor remains open so this context can fsync and authenticate the
        completed file before returning.
        """

        self._assert_open()
        if self._sealed:
            raise SecurePublicationError("Cannot add files after staging seal")
        name = _safe_component(name, "staging filename")
        if name not in self.expected_files:
            raise SecurePublicationError(f"Unexpected staging filename: {name!r}")
        if name in self._created:
            raise SecurePublicationError(f"Staging member already created: {name!r}")
        descriptor: int | None = None
        stream: io.FileIO | None = None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self._descriptor,
            )
            opened = os.fstat(descriptor)
            _require_regular_links(opened, f"staging member {name}")
            stream = os.fdopen(os.dup(descriptor), "wb", buffering=0)
            try:
                yield stream
            except BaseException:
                if not stream.closed:
                    stream.close()
                try:
                    selected = os.stat(
                        name, dir_fd=self._descriptor, follow_symlinks=False
                    )
                    if _same_inode(opened, selected):
                        _retire_owned_entry_at(
                            self._descriptor,
                            name,
                            selected,
                            f"failed staging member {name}",
                        )
                except (OSError, SecurePublicationError) as cleanup_error:
                    active = sys.exception()
                    if active is not None and hasattr(active, "add_note"):
                        active.add_note(
                            "Secure staging member retirement: "
                            f"{cleanup_error}"
                        )
                raise
            if not stream.closed:
                stream.close()
            _fsync_file(descriptor, f"staging member {name}")
            current = os.fstat(descriptor)
            selected = os.stat(
                name, dir_fd=self._descriptor, follow_symlinks=False
            )
            if (
                not _same_regular_inode(opened, current)
                or not _same_regular_inode(current, selected)
            ):
                raise SecurePublicationError(
                    f"Staging member identity changed: {name!r}"
                )
            self._created[name] = selected
        except SecurePublicationError:
            raise
        except OSError as exc:
            raise SecurePublicationError(
                f"Cannot create exclusive staging member {name!r}"
            ) from exc
        finally:
            if stream is not None and not stream.closed:
                stream.close()
            if descriptor is not None:
                os.close(descriptor)

    def snapshot(
        self,
        *,
        expected_hashes: Mapping[str, str] | None = None,
        file_validator: FileValidator | None = None,
        bundle_validator: BundleValidator | None = None,
    ) -> BundleSnapshot:
        """Seal and authenticate the exact staged closed set.

        Supplying independently computed hashes permits a metadata-only ctime
        refresh between the final writer close and this inspection.  The
        selected inode is reopened and rehashed before the refresh is
        accepted; byte changes still fail closed.
        """

        self._assert_open()
        if set(self._created) != set(self.expected_files):
            raise SecurePublicationError(
                "Not every expected staging member was created exclusively"
            )
        locked_hashes = (
            None
            if expected_hashes is None
            else _validate_expected_hashes(
                self.expected_files, expected_hashes
            )
        )
        snapshot, observed = _inspect_closed_directory_fd(
            self._descriptor,
            self.expected_files,
            "staging bundle",
            expected_hashes=locked_hashes,
            expected_stats=self._created,
            file_validator=file_validator,
            bundle_validator=bundle_validator,
            _permit_authenticated_ctime_refresh=True,
        )
        _fsync_directory(self._descriptor, "staging bundle after payloads")
        self._assert_open()
        final_snapshot, final_observed = _inspect_closed_directory_fd(
            self._descriptor,
            self.expected_files,
            "staging bundle after directory fsync",
            expected_hashes=snapshot.hashes,
            expected_stats={name: item.stat for name, item in observed.items()},
            _permit_authenticated_ctime_refresh=True,
        )
        if final_snapshot != snapshot:
            raise SecurePublicationError("Staging bundle changed while sealing")
        self._sealed_stats = {
            name: item.stat for name, item in final_observed.items()
        }
        self._sealed = True
        return final_snapshot

    def _read_existing(
        self,
        *,
        expected_hashes: Mapping[str, str],
        file_validator: FileValidator | None,
        bundle_validator: BundleValidator | None,
    ) -> tuple[
        BundleSnapshot,
        int,
        os.stat_result,
        dict[str, os.stat_result],
    ] | None:
        try:
            existing = os.stat(
                self.destination.name,
                dir_fd=self._parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
            raise SecurePublicationError(
                "Existing immutable destination is not a real directory"
            )
        descriptor, opened = _open_child_directory(
            self._parent,
            self.destination.name,
            "existing immutable destination",
        )
        try:
            snapshot, observed = _inspect_closed_directory_fd(
                descriptor,
                self.expected_files,
                "existing immutable destination",
                expected_hashes=expected_hashes,
                file_validator=file_validator,
                bundle_validator=bundle_validator,
                _permit_authenticated_ctime_refresh=True,
            )
            return (
                snapshot,
                descriptor,
                opened,
                {name: item.stat for name, item in observed.items()},
            )
        except Exception:
            os.close(descriptor)
            raise

    def publish(
        self,
        *,
        expected_hashes: Mapping[str, str],
        file_validator: FileValidator | None = None,
        bundle_validator: BundleValidator | None = None,
        pre_publish_gate: Gate | None = None,
        post_publish_gate: Gate | None = None,
    ) -> PublishedBundle:
        """Publish the sealed stage atomically, or reuse identical bytes.

        The caller must supply the hashes obtained from the semantic build's
        authenticated inventory.  Both callbacks may reject publication; the
        post callback is run before and after durability confirmation.
        """

        self._assert_open()
        if not self._sealed or self._sealed_stats is None:
            raise SecurePublicationError("Staging bundle must be sealed first")
        sealed_stats = self._sealed_stats
        locked_hashes = _validate_expected_hashes(
            self.expected_files, expected_hashes
        )
        pre_publish_gate = pre_publish_gate or (lambda: None)
        post_publish_gate = post_publish_gate or pre_publish_gate

        def confirm_existing(
            existing: tuple[
                BundleSnapshot,
                int,
                os.stat_result,
                dict[str, os.stat_result],
            ],
            kind: str,
        ) -> BundleSnapshot:
            existing_snapshot, descriptor, opened, member_stats = existing
            try:
                post_publish_gate()
                gate_snapshot, gate_observed = _inspect_closed_directory_fd(
                    descriptor,
                    self.expected_files,
                    f"{kind} final gate",
                    expected_hashes=locked_hashes,
                    expected_stats=member_stats,
                    file_validator=file_validator,
                    bundle_validator=bundle_validator,
                    _permit_authenticated_ctime_refresh=True,
                )
                member_stats = {
                    name: item.stat
                    for name, item in gate_observed.items()
                }
                current = os.stat(
                    self.destination.name,
                    dir_fd=self._parent.descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISDIR(current.st_mode) or not _same_inode(
                    opened, current
                ):
                    raise SecurePublicationError(f"{kind} identity changed")
                member_stats = _fsync_authenticated_members(
                    descriptor,
                    member_stats,
                    kind,
                    expected_hashes=locked_hashes,
                    _permit_authenticated_ctime_refresh=True,
                )
                _fsync_directory(descriptor, kind)
                _fsync_directory(
                    self._parent.descriptor,
                    f"immutable parent with {kind}",
                )
                post_publish_gate()
                final_snapshot, _ = _inspect_closed_directory_fd(
                    descriptor,
                    self.expected_files,
                    f"{kind} post-fsync gate",
                    expected_hashes=locked_hashes,
                    expected_stats=member_stats,
                    file_validator=file_validator,
                    bundle_validator=bundle_validator,
                    _permit_authenticated_ctime_refresh=True,
                )
                final_identity = os.stat(
                    self.destination.name,
                    dir_fd=self._parent.descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(opened, final_identity):
                    raise SecurePublicationError(
                        f"{kind} changed after final gate"
                    )
                _assert_directory_anchor(self._parent)
                if not (
                    existing_snapshot == gate_snapshot == final_snapshot
                ):
                    raise SecurePublicationError(f"{kind} snapshot changed")
                return final_snapshot
            finally:
                os.close(descriptor)

        def inspect_staging(label: str) -> BundleSnapshot:
            nonlocal sealed_stats
            stage_snapshot, observed = _inspect_closed_directory_fd(
                self._descriptor,
                self.expected_files,
                label,
                expected_hashes=locked_hashes,
                expected_stats=sealed_stats,
                file_validator=file_validator,
                bundle_validator=bundle_validator,
                _permit_authenticated_ctime_refresh=True,
            )
            sealed_stats = {
                name: item.stat for name, item in observed.items()
            }
            self._sealed_stats = sealed_stats
            return stage_snapshot

        def inspect_published(label: str) -> BundleSnapshot:
            nonlocal sealed_stats
            before_stats = sealed_stats
            stage_snapshot, observed = _inspect_closed_directory_fd(
                self._descriptor,
                self.expected_files,
                label,
                expected_hashes=locked_hashes,
                expected_stats=before_stats,
                file_validator=file_validator,
                bundle_validator=bundle_validator,
                _permit_authenticated_ctime_refresh=True,
            )
            observed_stats = {
                name: item.stat for name, item in observed.items()
            }
            refreshed = any(
                not _same_stable_file(before_stats[name], observed_stats[name])
                for name in self.expected_files
            )
            if refreshed:
                for durability_round in range(1, 4):
                    durable_stats = _fsync_authenticated_members(
                        self._descriptor,
                        observed_stats,
                        f"{label} ctime refresh durability",
                        expected_hashes=locked_hashes,
                        _permit_authenticated_ctime_refresh=True,
                    )
                    _fsync_directory(
                        self._descriptor,
                        f"{label} after ctime refresh durability "
                        f"round {durability_round}",
                    )
                    strict_snapshot, strict_observed = (
                        _inspect_closed_directory_fd(
                            self._descriptor,
                            self.expected_files,
                            f"{label} ctime-refresh durability gate "
                            f"round {durability_round}",
                            expected_hashes=locked_hashes,
                            expected_stats=durable_stats,
                            file_validator=file_validator,
                            bundle_validator=bundle_validator,
                            _permit_authenticated_ctime_refresh=True,
                        )
                    )
                    if strict_snapshot != stage_snapshot:
                        raise SecurePublicationError(
                            f"{label} snapshot changed during "
                            "ctime-refresh durability"
                        )
                    next_stats = {
                        name: item.stat
                        for name, item in strict_observed.items()
                    }
                    changed_again = any(
                        not _same_stable_file(
                            durable_stats[name], next_stats[name]
                        )
                        for name in self.expected_files
                    )
                    observed_stats = next_stats
                    if not changed_again:
                        break
                else:
                    raise SecurePublicationError(
                        f"{label} ctime did not stabilize after durability fsync"
                    )
            sealed_stats = observed_stats
            self._sealed_stats = sealed_stats
            return stage_snapshot

        snapshot = inspect_staging("staging bundle before publication")
        pre_publish_gate()
        self._assert_open()
        snapshot = inspect_staging("staging bundle at publication gate")
        existing = self._read_existing(
            expected_hashes=locked_hashes,
            file_validator=file_validator,
            bundle_validator=bundle_validator,
        )
        if existing is not None:
            existing_snapshot = confirm_existing(
                existing, "existing immutable destination"
            )
            self._discard_unpublished()
            return PublishedBundle(
                path=self.destination,
                snapshot=existing_snapshot,
                reused_existing=True,
            )

        self._assert_open()
        sealed_stats = _fsync_authenticated_members(
            self._descriptor,
            sealed_stats,
            "staging bundle",
            expected_hashes=locked_hashes,
            _permit_authenticated_ctime_refresh=True,
        )
        self._sealed_stats = sealed_stats
        self._assert_open()
        try:
            _rename_directory_noreplace_at(
                self._parent.descriptor,
                self._stage_name,
                self.destination.name,
            )
        except Exception as exc:
            try:
                destination_stat = os.stat(
                    self.destination.name,
                    dir_fd=self._parent.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                destination_stat = None
            if destination_stat is not None and _same_inode(
                self._opened, destination_stat
            ):
                # The native call may have completed before an injected or
                # transport-level error was reported.  This is our held stage,
                # not a reusable concurrent publication.
                self._published = True
            else:
                concurrent = self._read_existing(
                    expected_hashes=locked_hashes,
                    file_validator=file_validator,
                    bundle_validator=bundle_validator,
                )
                if concurrent is None:
                    raise SecurePublicationError(
                        "Cannot atomically no-replace the immutable staging "
                        "directory"
                    ) from exc
                concurrent_snapshot = confirm_existing(
                    concurrent, "concurrent immutable destination"
                )
                self._discard_unpublished()
                return PublishedBundle(
                    path=self.destination,
                    snapshot=concurrent_snapshot,
                    reused_existing=True,
                )
        else:
            self._published = True

        published = os.stat(
            self.destination.name,
            dir_fd=self._parent.descriptor,
            follow_symlinks=False,
        )
        if not _same_inode(self._opened, published):
            raise SecurePublicationError("Published immutable identity differs")
        post_publish_gate()
        published_snapshot = inspect_published(
            "published immutable destination"
        )
        _assert_directory_anchor(self._parent)
        _fsync_directory(self._descriptor, "published immutable destination")
        _fsync_directory(
            self._parent.descriptor,
            "immutable parent after atomic directory rename",
        )
        _assert_directory_anchor(self._parent)
        post_publish_gate()
        final_snapshot = inspect_published(
            "published immutable destination post-fsync gate"
        )
        final_stat = os.stat(
            self.destination.name,
            dir_fd=self._parent.descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_inode(self._opened, final_stat)
            or snapshot != published_snapshot
            or published_snapshot != final_snapshot
        ):
            raise SecurePublicationError(
                "Published immutable destination changed during commit"
            )
        return PublishedBundle(
            path=self.destination,
            snapshot=final_snapshot,
            reused_existing=False,
        )

    def _stage_name_for_inode(self) -> str | None:
        try:
            names = os.listdir(self._parent.descriptor)
        except OSError:
            return None
        for name in names:
            try:
                observed = os.stat(
                    name,
                    dir_fd=self._parent.descriptor,
                    follow_symlinks=False,
                )
            except OSError:
                continue
            if stat.S_ISDIR(observed.st_mode) and _same_inode(self._opened, observed):
                return name
        return None

    def _discard_unpublished(self) -> None:
        if self._published or self._closed or self._discarded:
            return
        selected = self._stage_name_for_inode()
        if selected is None:
            raise SecurePublicationError(
                "Owned staging directory path changed; stage quarantined"
            )
        try:
            _retire_owned_entry_at(
                self._parent.descriptor,
                selected,
                self._opened,
                "unpublished staging directory",
            )
        except (OSError, SecurePublicationError) as exc:
            raise SecurePublicationError(
                "Cannot safely retire authenticated unpublished staging directory; "
                "stage quarantined"
            ) from exc
        self._discarded = True

    def close(self) -> None:
        """Close anchored descriptors, discarding an unpublished stage."""

        if self._closed:
            return
        failure: BaseException | None = None
        try:
            self._discard_unpublished()
        except BaseException as exc:
            failure = exc
        finally:
            os.close(self._descriptor)
            self._closed = True
            _close_anchor(self._parent)
        if failure is not None:
            raise failure


def _read_optional_regular_at(
    parent: _DirectoryAnchor, name: str, label: str
) -> tuple[bytes, os.stat_result] | None:
    name = _safe_component(name, label)
    _assert_directory_anchor(parent)
    try:
        observed = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _require_regular_links(observed, label)
    payload = _read_regular_at(
        parent.descriptor, name, label, expected_stat=observed
    )
    after = os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
    if not _same_stable_file(observed, after):
        raise SecurePublicationError(f"Existing {label} identity changed")
    return payload, after


def _stat_optional_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _authenticate_payload_at(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    payload: bytes,
    label: str,
    *,
    expected_nlink: int = 1,
    fsync: bool = False,
    stable_expected: bool = False,
    _permit_authenticated_ctime_refresh: bool = False,
) -> os.stat_result:
    observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _require_regular_links(observed, label, expected_nlink=expected_nlink)
    if not _same_inode(expected, observed):
        raise SecurePublicationError(f"{label} owned inode changed")
    if stable_expected and not _same_stable_regular(
        expected, observed, expected_nlink=expected_nlink
    ):
        may_reauthenticate_ctime = (
            _permit_authenticated_ctime_refresh
            and expected_nlink == 1
            and expected.st_ctime_ns != observed.st_ctime_ns
            and _same_stable_file_except_ctime(expected, observed)
        )
        if not may_reauthenticate_ctime:
            raise SecurePublicationError(f"{label} stable identity changed")
    actual = _read_regular_at(
        directory_fd,
        name,
        label,
        expected_nlink=expected_nlink,
        expected_stat=observed,
        fsync=fsync,
    )
    if actual != payload:
        raise SecurePublicationError(f"{label} bytes changed")
    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not _same_stable_regular(
        observed, after, expected_nlink=expected_nlink
    ):
        raise SecurePublicationError(f"{label} changed during authentication")
    return after


def _matches_owned_entry(
    expected: os.stat_result,
    observed: os.stat_result | None,
    *,
    expected_nlink: int = 1,
) -> bool:
    """Return whether ``observed`` is the same supported owned entry."""

    if observed is None or not _same_inode(expected, observed):
        return False
    if stat.S_ISREG(expected.st_mode):
        return _same_regular_inode(
            expected, observed, expected_nlink=expected_nlink
        )
    return stat.S_ISDIR(expected.st_mode) and stat.S_ISDIR(observed.st_mode)


def _retire_owned_entry_at(
    directory_fd: int,
    name: str,
    expected: os.stat_result,
    label: str,
    *,
    expected_nlink: int = 1,
) -> str:
    """Atomically remove an owned name from service without unlinking an inode.

    POSIX has no portable unlink-if-inode-matches operation.  A check followed
    by ``unlink`` can therefore delete a concurrent authority that takes over
    the basename between those operations.  Retirement uses native
    ``RENAME_NOREPLACE`` instead.  The owned entry is moved to a random recovery
    basename and retained.  If an unowned entry is displaced from ``name``, it
    is restored without replacement; no unknown inode is ever deleted.
    """

    name = _safe_component(name, label)
    selected = _stat_optional_at(directory_fd, name)
    if selected is None:
        raise SecurePublicationError(f"{label} disappeared before owned retirement")
    if stat.S_ISREG(expected.st_mode):
        _require_regular_links(selected, label, expected_nlink=expected_nlink)
    elif not stat.S_ISDIR(expected.st_mode) or not stat.S_ISDIR(selected.st_mode):
        raise SecurePublicationError(f"Unsupported owned-entry type for {label}")
    if not _matches_owned_entry(
        expected, selected, expected_nlink=expected_nlink
    ):
        raise SecurePublicationError(
            f"{label} retirement conflict; unowned entry preserved"
        )

    retired_name = f".{name}.{secrets.token_hex(16)}.retired"
    rename_error: BaseException | None = None
    try:
        _rename_noreplace_at(
            directory_fd,
            name,
            directory_fd,
            retired_name,
        )
    except BaseException as exc:
        rename_error = exc

    source_after = _stat_optional_at(directory_fd, name)
    retired_after = _stat_optional_at(directory_fd, retired_name)
    if source_after is None and _matches_owned_entry(
        expected, retired_after, expected_nlink=expected_nlink
    ):
        _fsync_directory(directory_fd, f"{label} parent after owned retirement")
        return retired_name

    if _matches_owned_entry(
        expected, source_after, expected_nlink=expected_nlink
    ) and retired_after is None:
        raise SecurePublicationError(
            f"{label} retirement did not take effect; owned entry preserved"
        ) from rename_error

    # A last-window substitution may have been moved to the recovery basename.
    # Restore it without replacing any newer entry at the authoritative name.
    if source_after is None and retired_after is not None:
        restore_error: BaseException | None = None
        try:
            _rename_noreplace_at(
                directory_fd,
                retired_name,
                directory_fd,
                name,
            )
        except BaseException as exc:
            restore_error = exc
        source_restored = _stat_optional_at(directory_fd, name)
        retired_restored = _stat_optional_at(directory_fd, retired_name)
        if (
            retired_restored is None
            and source_restored is not None
            and _same_inode(retired_after, source_restored)
        ):
            _fsync_directory(
                directory_fd, f"{label} parent after concurrent-entry restoration"
            )
            raise SecurePublicationError(
                f"{label} retirement conflict; concurrent entry restored"
            ) from rename_error
        raise SecurePublicationError(
            f"{label} retirement conflict needs manual recovery; all entries preserved"
        ) from restore_error

    raise SecurePublicationError(
        f"{label} retirement layout is ambiguous; all entries preserved"
    ) from rename_error


def replace_pointer_last(
    path: Path,
    payload: bytes,
    *,
    trusted_root: Path,
    pre_replace_gate: Gate,
    post_replace_gate: Gate | None = None,
    label: str = "authority pointer",
) -> bool:
    """Install ``path`` with atomic compare-and-swap filesystem semantics.

    An absent pointer is installed with native ``RENAME_NOREPLACE``.  An
    existing pointer is replaced with native ``RENAME_EXCHANGE`` so the exact
    displaced inode becomes the rollback slot.  Unknown or concurrent entries
    are never overwritten or unlinked.  Obsolete owned entries are moved to
    random ``.retired`` recovery names because POSIX has no portable atomic
    unlink-if-inode-matches primitive.
    """

    if not isinstance(payload, bytes):
        raise SecurePublicationError("Pointer payload must be bytes")
    if not callable(pre_replace_gate):
        raise SecurePublicationError("Pointer pre-replace gate must be callable")
    post_replace_gate = post_replace_gate or pre_replace_gate
    root = _lexical_absolute(trusted_root)
    pointer = _lexical_absolute(path)
    relative = _relative_beneath(pointer, root, label)
    if not relative.parts:
        raise SecurePublicationError("Pointer cannot equal trusted root")
    name = _safe_component(pointer.name, label)
    parent = _open_directory_anchor(
        pointer.parent,
        f"{label} parent",
        create=True,
        trusted_root=root,
    )
    transaction_name = f".{name}.{secrets.token_hex(16)}.transaction"
    candidate_stat: os.stat_result | None = None
    installed_stat: os.stat_result | None = None
    recovery_stat: os.stat_result | None = None
    old_payload: bytes | None = None
    transaction_role: str | None = None
    pointer_replaced = False
    preserve_transaction = False

    def fail(message: str, cause: BaseException | None = None) -> None:
        error = SecurePublicationError(message)
        if cause is None:
            raise error
        raise error from cause

    def pointer_equals_expected(expected: os.stat_result) -> os.stat_result:
        selected = _read_optional_regular_at(parent, name, label)
        if selected is None or selected[0] != payload:
            raise SecurePublicationError(
                f"{label} bytes changed during final commit gate"
            )
        if not _same_stable_file(expected, selected[1]):
            raise SecurePublicationError(
                f"{label} stable identity changed during final commit gate"
            )
        return selected[1]

    def same_candidate(observed: os.stat_result | None) -> bool:
        return (
            candidate_stat is not None
            and observed is not None
            and _same_regular_inode(candidate_stat, observed)
        )

    def same_recovery(observed: os.stat_result | None) -> bool:
        return (
            recovery_stat is not None
            and observed is not None
            and _same_regular_inode(recovery_stat, observed)
        )

    def rollback() -> None:
        nonlocal pointer_replaced, transaction_role, preserve_transaction
        nonlocal recovery_stat
        if not pointer_replaced or installed_stat is None:
            return
        try:
            candidate = _authenticate_payload_at(
                parent.descriptor,
                name,
                installed_stat,
                payload,
                f"installed {label} before rollback",
                stable_expected=True,
            )
        except Exception as exc:
            preserve_transaction = transaction_role == "recovery"
            raise SecurePublicationError(
                f"{label} rollback conflict; concurrent authority preserved"
            ) from exc

        if transaction_role is None:
            try:
                _retire_owned_entry_at(
                    parent.descriptor,
                    name,
                    candidate,
                    f"newly installed {label} rollback entry",
                )
                pointer_replaced = False
            except Exception as exc:
                raise SecurePublicationError(
                    f"Cannot safely retire newly installed {label} during rollback"
                ) from exc
            return

        if (
            transaction_role != "recovery"
            or recovery_stat is None
            or old_payload is None
        ):
            preserve_transaction = True
            raise SecurePublicationError(
                f"{label} rollback slot is unauthenticated; entries preserved"
            )
        try:
            old_recovery = _authenticate_payload_at(
                parent.descriptor,
                transaction_name,
                recovery_stat,
                old_payload,
                f"{label} rollback slot",
                fsync=True,
                stable_expected=True,
            )
        except Exception as exc:
            preserve_transaction = True
            raise SecurePublicationError(
                f"{label} rollback slot changed; entries preserved"
            ) from exc

        exchange_error: BaseException | None = None
        try:
            _rename_exchange_at(parent.descriptor, transaction_name, name)
        except BaseException as exc:
            exchange_error = exc
        transaction_after = _stat_optional_at(parent.descriptor, transaction_name)
        pointer_after = _stat_optional_at(parent.descriptor, name)

        if same_candidate(transaction_after) and _same_regular_inode(
            old_recovery, pointer_after
        ):
            transaction_role = "candidate"
            pointer_replaced = False
        elif _same_regular_inode(old_recovery, transaction_after) and same_candidate(
            pointer_after
        ):
            fail(f"{label} rollback exchange did not take effect", exchange_error)
        elif pointer_after is not None and _same_regular_inode(
            old_recovery, pointer_after
        ) and transaction_after is not None:
            # A concurrent authority occupied the pointer immediately before the
            # exchange and was displaced into the rollback slot.  Reverse once.
            displaced = transaction_after
            reverse_error: BaseException | None = None
            try:
                _rename_exchange_at(parent.descriptor, transaction_name, name)
            except BaseException as exc:
                reverse_error = exc
            transaction_restored = _stat_optional_at(
                parent.descriptor, transaction_name
            )
            pointer_restored = _stat_optional_at(parent.descriptor, name)
            if _same_regular_inode(
                old_recovery, transaction_restored
            ) and _same_inode(displaced, pointer_restored):
                _fsync_directory(
                    parent.descriptor,
                    f"{label} parent after rollback conflict reversal",
                )
                preserve_transaction = True
                raise SecurePublicationError(
                    f"{label} rollback CAS conflict; concurrent authority restored"
                ) from exchange_error
            preserve_transaction = True
            fail(
                f"{label} rollback reversal needs manual recovery; entries preserved",
                reverse_error,
            )
        else:
            preserve_transaction = True
            fail(
                f"{label} rollback layout is ambiguous; entries preserved",
                exchange_error,
            )

        restored = _authenticate_payload_at(
            parent.descriptor,
            name,
            old_recovery,
            old_payload,
            f"restored {label}",
            fsync=True,
        )
        displaced_candidate = _authenticate_payload_at(
            parent.descriptor,
            transaction_name,
            candidate,
            payload,
            f"displaced {label} rollback candidate",
        )
        _fsync_directory(parent.descriptor, f"{label} parent after rollback")
        try:
            _retire_owned_entry_at(
                parent.descriptor,
                transaction_name,
                displaced_candidate,
                f"displaced {label} rollback candidate",
            )
            transaction_role = None
            recovery_stat = restored
        except Exception as exc:
            preserve_transaction = True
            raise SecurePublicationError(
                f"Restored {label} is safe but rollback retirement needs review"
            ) from exc

    try:
        initial = _read_optional_regular_at(parent, name, label)
        if initial is not None and initial[0] == payload:
            pre_replace_gate()
            pointer_equals_expected(initial[1])
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent.descriptor,
            )
            try:
                selected = os.fstat(descriptor)
                if not _same_stable_file(initial[1], selected):
                    raise SecurePublicationError(
                        f"Existing {label} identity changed before fsync"
                    )
                _fsync_file(descriptor, f"existing {label}")
                after_fsync = os.fstat(descriptor)
                after_path = os.stat(
                    name,
                    dir_fd=parent.descriptor,
                    follow_symlinks=False,
                )
                if (
                    not _same_stable_file(selected, after_fsync)
                    or not _same_stable_file(after_fsync, after_path)
                ):
                    raise SecurePublicationError(
                        f"Existing {label} changed during fsync"
                    )
            finally:
                os.close(descriptor)
            _fsync_directory(
                parent.descriptor,
                f"{label} parent after idempotent durability confirmation",
            )
            post_replace_gate()
            pointer_equals_expected(initial[1])
            _assert_directory_anchor(parent)
            return False

        candidate_stat = _write_payload_at(
            parent.descriptor,
            transaction_name,
            payload,
            f"{label} temporary",
        )
        transaction_role = "candidate"
        pre_replace_gate()
        _assert_directory_anchor(parent)
        existing = _read_optional_regular_at(parent, name, label)
        if (initial is None) != (existing is None) or (
            initial is not None
            and existing is not None
            and not _same_stable_file(initial[1], existing[1])
        ):
            raise SecurePublicationError(
                f"{label} changed after initial observation; concurrent authority preserved"
            )
        anchored_candidate = _authenticate_payload_at(
            parent.descriptor,
            transaction_name,
            candidate_stat,
            payload,
            f"{label} transaction candidate before install",
            fsync=True,
            stable_expected=True,
            _permit_authenticated_ctime_refresh=True,
        )
        candidate_stat = anchored_candidate

        install_error: BaseException | None = None
        if existing is None:
            try:
                _rename_noreplace_at(
                    parent.descriptor,
                    transaction_name,
                    parent.descriptor,
                    name,
                )
            except BaseException as exc:
                install_error = exc
            transaction_after = _stat_optional_at(
                parent.descriptor, transaction_name
            )
            pointer_after = _stat_optional_at(parent.descriptor, name)
            if transaction_after is None and same_candidate(pointer_after):
                transaction_role = None
                pointer_replaced = True
            elif same_candidate(transaction_after) and pointer_after is None:
                fail(f"{label} no-replace install did not take effect", install_error)
            elif same_candidate(transaction_after) and pointer_after is not None:
                raise SecurePublicationError(
                    f"{label} no-replace conflict; concurrent authority preserved"
                ) from install_error
            else:
                preserve_transaction = transaction_after is not None
                fail(
                    f"{label} no-replace install layout is ambiguous; entries preserved",
                    install_error,
                )
        else:
            old_payload = existing[0]
            expected_old = existing[1]
            try:
                _rename_exchange_at(parent.descriptor, transaction_name, name)
            except BaseException as exc:
                install_error = exc
            transaction_after = _stat_optional_at(
                parent.descriptor, transaction_name
            )
            pointer_after = _stat_optional_at(parent.descriptor, name)
            if _same_regular_inode(
                expected_old, transaction_after
            ) and same_candidate(pointer_after):
                recovery_stat = transaction_after
                transaction_role = "recovery"
                pointer_replaced = True
            elif same_candidate(transaction_after) and _same_regular_inode(
                expected_old, pointer_after
            ):
                fail(f"{label} exchange install did not take effect", install_error)
            elif same_candidate(pointer_after) and transaction_after is not None:
                displaced = transaction_after
                reverse_error: BaseException | None = None
                try:
                    _rename_exchange_at(
                        parent.descriptor, transaction_name, name
                    )
                except BaseException as exc:
                    reverse_error = exc
                transaction_restored = _stat_optional_at(
                    parent.descriptor, transaction_name
                )
                pointer_restored = _stat_optional_at(parent.descriptor, name)
                if same_candidate(transaction_restored) and _same_inode(
                    displaced, pointer_restored
                ):
                    _fsync_directory(
                        parent.descriptor,
                        f"{label} parent after install conflict reversal",
                    )
                    transaction_role = "candidate"
                    raise SecurePublicationError(
                        f"{label} exchange conflict; concurrent authority restored"
                    ) from install_error
                preserve_transaction = True
                fail(
                    f"{label} install reversal needs manual recovery; entries preserved",
                    reverse_error,
                )
            else:
                preserve_transaction = transaction_after is not None
                fail(
                    f"{label} exchange install layout is ambiguous; entries preserved",
                    install_error,
                )

        installed_stat = _authenticate_payload_at(
            parent.descriptor,
            name,
            candidate_stat,
            payload,
            f"published {label}",
        )
        if transaction_role == "recovery":
            assert recovery_stat is not None and old_payload is not None
            recovery_stat = _authenticate_payload_at(
                parent.descriptor,
                transaction_name,
                recovery_stat,
                old_payload,
                f"{label} rollback slot after install",
                fsync=True,
            )
        try:
            pointer_equals_expected(installed_stat)
            _assert_directory_anchor(parent)
            post_replace_gate()
            pointer_equals_expected(installed_stat)
            _assert_directory_anchor(parent)
            _fsync_directory(
                parent.descriptor, f"{label} parent after pointer-last rename"
            )
            _assert_directory_anchor(parent)
            post_replace_gate()
            pointer_equals_expected(installed_stat)
            if transaction_role == "recovery":
                assert recovery_stat is not None and old_payload is not None
                recovery_stat = _authenticate_payload_at(
                    parent.descriptor,
                    transaction_name,
                    recovery_stat,
                    old_payload,
                    f"{label} rollback slot at final gate",
                    fsync=True,
                    stable_expected=True,
                )
        except BaseException:
            try:
                rollback()
            except BaseException as rollback_error:
                raise SecurePublicationError(
                    f"{label} commit failed and rollback needs manual recovery"
                ) from rollback_error
            raise

        if transaction_role == "recovery":
            assert recovery_stat is not None and old_payload is not None
            try:
                authenticated_recovery = _authenticate_payload_at(
                    parent.descriptor,
                    transaction_name,
                    recovery_stat,
                    old_payload,
                    f"{label} old authority before retirement",
                    stable_expected=True,
                )
                _retire_owned_entry_at(
                    parent.descriptor,
                    transaction_name,
                    authenticated_recovery,
                    f"{label} old authority after successful commit",
                )
                transaction_role = None
            except Exception as exc:
                preserve_transaction = True
                raise SecurePublicationError(
                    f"Committed {label}; old-authority retirement needs review"
                ) from exc
        return True
    finally:
        cleanup_failure: BaseException | None = None
        transaction = _stat_optional_at(parent.descriptor, transaction_name)
        if transaction is not None and not preserve_transaction:
            try:
                if transaction_role != "candidate" or candidate_stat is None:
                    raise SecurePublicationError(
                        f"Unknown {label} transaction entry preserved"
                    )
                authenticated_candidate = _authenticate_payload_at(
                    parent.descriptor,
                    transaction_name,
                    candidate_stat,
                    payload,
                    f"{label} aborted transaction candidate",
                )
                _retire_owned_entry_at(
                    parent.descriptor,
                    transaction_name,
                    authenticated_candidate,
                    f"{label} aborted transaction candidate",
                )
                transaction_role = None
            except BaseException as exc:
                cleanup_failure = SecurePublicationError(
                    f"{label} transaction retirement conflict; entry preserved"
                )
                cleanup_failure.__cause__ = exc
        _close_anchor(parent)
        if cleanup_failure is not None:
            raise cleanup_failure


__all__ = [
    "AuthenticatedBundleGuard",
    "AuthenticatedPointerGuard",
    "BundleSnapshot",
    "BundleValidator",
    "FileRecord",
    "FileValidator",
    "Gate",
    "PayloadValidator",
    "PublishedBundle",
    "SecurePublicationError",
    "SecureStagingDirectory",
    "replace_pointer_last",
]
