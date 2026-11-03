#!/usr/bin/env python3
"""Write a reviewed binary manifest for the VASP port's managed runner.

Contract: docs/CONTRACT-v0.3.md, section C. Reads a relink directory made by
scripts/relink-vasp.sh, the port's effective toolchain and the backend build
identity, and writes PORT_ROOT/private/binary-manifests/NAME.json exclusively.
Nothing is executed except `otool -L` and `file` on the new executable and the
port's own toolchain reader. Writing the file is the review step: inspect it
before use, and never edit it afterwards (write a new name instead).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z')
TOOLCHAIN_KEYS = ('PORT_GCC_VERSION', 'PORT_OPENMPI_VERSION', 'PORT_PRRTE_VERSION',
                  'PORT_OPENBLAS_VERSION', 'PORT_SCALAPACK_VERSION', 'PORT_FFTW_VERSION')


def sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def runtime_dependencies(executable):
    """Homebrew libraries the executable loads, pinned by load path and bytes."""
    listing = subprocess.check_output(['/usr/bin/otool', '-L', str(executable)], text=True)
    deps = []
    for line in listing.splitlines()[1:]:
        path = line.strip().split(' (')[0]
        if path.startswith('/opt/homebrew/'):
            resolved = os.path.realpath(path)
            deps.append({'path': path, 'resolved': resolved, 'sha256': sha256(resolved)})
    return deps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--port-root', required=True, type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--relink-dir', required=True, type=Path, help='output of relink-vasp.sh')
    p.add_argument('--mode', required=True, choices=('off', 'auto', 'force'))
    p.add_argument('--check', required=True, choices=('0', '1'))
    p.add_argument('--aggregator', required=True, type=Path,
                   help='installed copy of scripts/vmb_evidence.py the runner will execute')
    p.add_argument('--source-commit', required=True)
    p.add_argument('--source-digest', required=True)
    p.add_argument('--source-dirty', required=True, choices=('0', '1'))
    args = p.parse_args(argv)
    if not NAME.fullmatch(args.name):
        p.error('--name must be 1-64 characters of [A-Za-z0-9._-]')
    port = args.port_root.resolve(strict=True)
    relink = args.relink_dir.resolve(strict=True)
    executable = relink / 'vasp_std_vmb'
    info = dict(line.split('=', 1) for line in (relink / 'relink.txt').read_text().splitlines())
    toolchain = dict(line.split('=', 1) for line in subprocess.check_output(
        ['/usr/bin/python3', '-I', '-B', str(port / 'scripts/toolchain.py')], text=True).splitlines()
        if '=' in line and not line.startswith('#'))
    for key in TOOLCHAIN_KEYS:
        if key != 'PORT_PRRTE_VERSION' and info.get(key) != toolchain.get(key):
            p.error(f'{key} of the relink ({info.get(key)}) differs from the port ({toolchain.get(key)})')
    description = subprocess.check_output(['/usr/bin/file', '-b', str(executable)], text=True)
    if 'Mach-O' not in description or 'arm64' not in description:
        p.error('executable is not an arm64 Mach-O file')
    aggregator = args.aggregator.resolve(strict=True)
    manifest = {
        'schema': 'vasp-port.binary-manifest.v1', 'name': args.name, 'status': 'reviewed',
        'executable': {'path': str(executable), 'sha256': sha256(executable),
                       'bytes': executable.stat().st_size},
        'vasp': {'version': '6.6.1', 'variant': 'std'},
        'backend': {'name': 'vasp-metal-backend', 'version': '0.3.0',
                    'source_commit': args.source_commit, 'source_digest': args.source_digest,
                    'source_dirty': args.source_dirty == '1',
                    'library_sha256': info['library_sha256'], 'report_schema': 'vmb.rank-report.v1',
                    'aggregator': {'path': str(aggregator), 'sha256': sha256(aggregator)}},
        'build': {'objects': info['objects'], 'object_count': int(info['object_count']),
                  'relink_script_sha256': info['relink_script_sha256'],
                  'link_log_sha256': sha256(relink / 'link.log'), 'built_utc': info['built_utc'],
                  'toolchain_source': info['toolchain_source']},
        'toolchain': {key: toolchain[key] for key in TOOLCHAIN_KEYS},
        'runtime_dependencies': runtime_dependencies(executable),
        'selection': {'mode': args.mode, 'shadow_check': args.check == '1'},
        'written_utc': datetime.now(timezone.utc).isoformat(),
    }
    directory = port / 'private/binary-manifests'
    directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    target = directory / (args.name + '.json')
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print(target)
    return 0


if __name__ == '__main__':
    sys.exit(main())
