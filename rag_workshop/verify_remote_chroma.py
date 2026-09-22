#!/usr/bin/env python3
"""Compare a local Chroma collection with the collection exposed through an SSH tunnel."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a local Chroma collection is available remotely."
    )
    parser.add_argument(
        "--local-db",
        type=Path,
        default=Path("rag_workshop/chroma_db"),
        help="Local Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default="ccin2p3_docs",
        help="Collection name to compare.",
    )
    parser.add_argument(
        "--remote-host",
        default="127.0.0.1",
        help="Remote Chroma host as seen from the laptop, normally 127.0.0.1.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=8000,
        help="Local forwarded Chroma port.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of IDs to compare as a sample.",
    )
    parser.add_argument("--remote-db", type=Path, help="Remote Chroma directory for hash fallback.")
    parser.add_argument("--remote-user", default="labobots")
    parser.add_argument("--ssh-host", default="195.221.220.18")
    parser.add_argument("--ssh-port", type=int, default=22003)
    parser.add_argument(
        "--all-ids",
        action="store_true",
        help="Compare every collection ID; can be expensive for large collections.",
    )
    return parser.parse_args()


def collection_ids(collection, limit: int | None = None) -> set[str]:
    result = collection.get(limit=limit, include=[])
    return set(result["ids"])


def local_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[path.relative_to(root).as_posix()] = digest
    return manifest


def remote_manifest(root: Path, args: argparse.Namespace) -> dict[str, str]:
    target = f"{args.remote_user}@{args.ssh_host}"
    command = (
        f"cd {shlex.quote(str(root))} && "
        "find . -type f -print0 | sort -z | xargs -0 sha256sum"
    )
    result = subprocess.run(
        ["ssh", "-p", str(args.ssh_port), target, command],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = {}
    for line in result.stdout.splitlines():
        digest, relative = line.split(None, 1)
        manifest[relative.removeprefix("./")] = digest
    return manifest


def verify_file_fallback(args: argparse.Namespace) -> int:
    if args.remote_db is None:
        print(
            "chromadb is not installed locally. Install it or provide --remote-db "
            "for the file-hash fallback.",
            file=sys.stderr,
        )
        return 2
    print("chromadb is not installed locally; using a SHA-256 database-file comparison.")
    try:
        local = local_manifest(args.local_db)
        remote = remote_manifest(args.remote_db, args)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not build local/remote manifests: {exc}", file=sys.stderr)
        return 1
    if local != remote:
        print("FAIL: local and remote Chroma file manifests differ.", file=sys.stderr)
        only_local = sorted(set(local) - set(remote))
        only_remote = sorted(set(remote) - set(local))
        changed = sorted(name for name in set(local) & set(remote) if local[name] != remote[name])
        print("Only local:", only_local[:10], file=sys.stderr)
        print("Only remote:", only_remote[:10], file=sys.stderr)
        print("Changed files:", changed[:10], file=sys.stderr)
        print(
            "Note: binary Chroma hashes can change after the server opens the database; "
            "use the API-based check for collection-level verification.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: all {len(local)} Chroma files have matching SHA-256 hashes.")
    return 0


def main() -> int:
    args = parse_args()
    if not args.local_db.is_dir():
        print(f"Local Chroma directory not found: {args.local_db}", file=sys.stderr)
        return 2

    try:
        import chromadb
    except ModuleNotFoundError:
        return verify_file_fallback(args)

    try:
        local_client = chromadb.PersistentClient(path=str(args.local_db))
        remote_client = chromadb.HttpClient(
            host=args.remote_host,
            port=args.remote_port,
        )
        print("Remote heartbeat:", remote_client.heartbeat())
        local_collection = local_client.get_collection(args.collection)
        remote_collection = remote_client.get_collection(args.collection)
    except Exception as exc:
        print(f"Connection or collection error: {exc}", file=sys.stderr)
        print(
            "Check that the SSH tunnel is running and forwards local port "
            f"{args.remote_port} to remote Chroma.",
            file=sys.stderr,
        )
        return 1

    local_count = local_collection.count()
    remote_count = remote_collection.count()
    print(f"Collection: {args.collection}")
    print(f"Local documents : {local_count}")
    print(f"Remote documents: {remote_count}")

    if local_count != remote_count:
        print("FAIL: document counts differ.", file=sys.stderr)
        return 1

    if args.all_ids:
        local_ids = collection_ids(local_collection)
        remote_ids = collection_ids(remote_collection)
        if local_ids != remote_ids:
            print(
                f"FAIL: ID sets differ (local={len(local_ids)}, remote={len(remote_ids)}).",
                file=sys.stderr,
            )
            print("Only local:", sorted(local_ids - remote_ids)[:10], file=sys.stderr)
            print("Only remote:", sorted(remote_ids - local_ids)[:10], file=sys.stderr)
            return 1
        print(f"All IDs match: {len(local_ids)} IDs checked.")
    else:
        sample_size = min(args.sample_size, local_count)
        local_ids = collection_ids(local_collection, limit=sample_size)
        remote_ids = collection_ids(remote_collection, limit=sample_size)
        print(f"Local sample IDs : {sorted(local_ids)}")
        print(f"Remote sample IDs: {sorted(remote_ids)}")
        if local_ids != remote_ids:
            print("FAIL: sample ID sets differ.", file=sys.stderr)
            return 1
        print(f"Sample IDs match: {sample_size} IDs checked.")

    print("OK: local and remote Chroma collections are consistent for the checks performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
