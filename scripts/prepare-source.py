#!/usr/bin/env python3
"""Fetch the pinned upstream archive, without model weights or SDK files."""
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release-manifest.json').read_text())
commit=manifest['sources']['llama_cpp_commit']
destination=root/'build-inputs/llama.cpp'
if destination.exists():
    raise SystemExit('build-inputs/llama.cpp already exists; verify it or remove it explicitly before refetching')
with tempfile.TemporaryDirectory() as tmp:
    archive=Path(tmp)/'source.tar.gz'
    url=f'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{commit}'
    with urllib.request.urlopen(url,timeout=120) as response,archive.open('wb') as output:
        shutil.copyfileobj(response,output)
    extract=Path(tmp)/'source'; extract.mkdir()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            target=(extract/member.name).resolve()
            if not target.is_relative_to(extract.resolve()) or member.isdev():
                raise ValueError('Unsafe archive member')
            if member.issym():
                if not (target.parent/member.linkname).resolve().is_relative_to(extract.resolve()):
                    raise ValueError('Unsafe archive symlink')
            if member.islnk():
                if not (extract/member.linkname).resolve().is_relative_to(extract.resolve()):
                    raise ValueError('Unsafe archive hardlink')
        stream.extractall(extract)
    directories=list(extract.iterdir())
    if len(directories)!=1 or not directories[0].is_dir():
        raise ValueError('Unexpected archive layout')
    subprocess.run([sys.executable,str(root/'scripts/source-tree-sha256.py'),str(directories[0]),'--manifest',str(root/'release-manifest.json')],check=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(directories[0],destination,symlinks=True)
print(destination)
