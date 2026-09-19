#!/usr/bin/env python3
"""Normalize the known cache-fix ELF search path without moving any ELF offsets.

This is deliberately not a general ELF editor. Only the original checksum-pinned
artifact, or its already-sanitized counterpart, is accepted. Numerical/runtime
acceptance of the sanitized library is performed separately on the target NPU.
"""

import argparse
import hashlib
from pathlib import Path
import struct


ORIGINAL_SHA256 = "3a6ed3baeaed2ff69ddfc264461433352697e64c7463be52caf41a26cf4bf9da"
SANITIZED_SHA256 = "344f759c370ab28b08ce38d73aa2e2b464a5ce84a139a7d726704e879570c324"
RUNTIME_PATH = b"/usr/local/Ascend/cann/lib64"


def search_path(blob: bytes) -> tuple[int, bytes]:
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian ELF64 file")
    if struct.unpack_from("<H", blob, 18)[0] != 183:
        raise ValueError("Expected AArch64 ELF machine type")
    section_offset = struct.unpack_from("<Q", blob, 40)[0]
    entry_bytes, section_count = struct.unpack_from("<HH", blob, 58)
    if entry_bytes != 64 or section_offset + section_count * entry_bytes > len(blob):
        raise ValueError("Invalid ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", blob, section_offset + i * entry_bytes)
                for i in range(section_count)]
    found: list[tuple[int, bytes]] = []
    for section in sections:
        if section[1] != 6:  # SHT_DYNAMIC
            continue
        if section[6] >= len(sections) or section[9] != 16:
            raise ValueError("Invalid dynamic section")
        strings = sections[section[6]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            tag, value = struct.unpack_from("<QQ", blob, offset)
            if tag not in (15, 29):  # DT_RPATH or DT_RUNPATH
                continue
            start = strings[4] + value
            end = blob.find(b"\0", start, strings[4] + strings[5])
            if start < strings[4] or end < start:
                raise ValueError("Invalid dynamic search-path string")
            found.append((start, blob[start:end]))
    if len(found) != 1:
        raise ValueError("Expected exactly one ELF dynamic search-path entry")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path, nargs="?", default=
                        Path(__file__).resolve().parents[1] / "artifacts/cache-fix/libnnopbase.so")
    parser.add_argument("--check", action="store_true", help="Verify an already-sanitized library without writing")
    args = parser.parse_args()
    blob = args.library.read_bytes()
    original_hash = hashlib.sha256(blob).hexdigest()
    if original_hash not in (ORIGINAL_SHA256, SANITIZED_SHA256):
        raise ValueError("Unrecognized cache-fix library checksum; refusing modification")
    start, old = search_path(blob)
    if original_hash == SANITIZED_SHA256:
        if old != RUNTIME_PATH:
            raise ValueError("Sanitized checksum has an unexpected ELF search path")
        print(f"already_sanitized sha256={SANITIZED_SHA256} rpath={RUNTIME_PATH.decode()}")
        return
    if args.check:
        raise ValueError("The recognized original library has not been sanitized")
    if blob.count(old) != 1 or len(old) != 90 or not old.startswith(b"/work/"):
        raise ValueError("Original search-path occurrence/layout differs from the verified artifact")
    if not old.endswith(b":" + RUNTIME_PATH + b":") or old.count(b":") != 2:
        raise ValueError("Original path must contain the build directory, runtime directory and final empty item")
    replacement = RUNTIME_PATH + b"\0" * (len(old) - len(RUNTIME_PATH))
    patched = blob[:start] + replacement + blob[start + len(old):]
    if len(patched) != len(blob) or hashlib.sha256(patched).hexdigest() != SANITIZED_SHA256:
        raise ValueError("Sanitized library checksum/size mismatch")
    if search_path(patched)[1] != RUNTIME_PATH:
        raise ValueError("ELF path verification failed after transformation")
    # Identical-length replacement preserves every section/segment/symbol offset.
    args.library.write_bytes(patched)
    if hashlib.sha256(args.library.read_bytes()).hexdigest() != SANITIZED_SHA256:
        raise ValueError("Written artifact checksum mismatch")
    print(f"sanitized sha256={SANITIZED_SHA256} rpath={RUNTIME_PATH.decode()} bytes_changed_at={start}")


if __name__ == "__main__":
    main()
