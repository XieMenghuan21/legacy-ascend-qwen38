"""Hash a staged source tree deterministically; optionally enforce its manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def source_digest(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"Missing source tree: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            kind = b"symlink"
            payload = str(path.readlink()).encode("utf-8")
        elif path.is_file():
            kind = b"file"
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(block)
            payload = file_digest.digest()
        else:
            continue
        digest.update(kind + b"\0" + relative.as_posix().encode("utf-8") + b"\0" + payload + b"\0")
        count += 1
    if count == 0:
        raise ValueError("Source tree has no files")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    actual = source_digest(args.root)
    if args.manifest:
        expected = json.loads(args.manifest.read_text())["sources"].get("llama_cpp_source_tree_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Manifest needs a verified llama_cpp_source_tree_sha256 before building")
        if actual != expected:
            raise ValueError(f"Source tree digest mismatch: expected {expected}, actual {actual}")
    print(actual)


if __name__ == "__main__":
    main()
