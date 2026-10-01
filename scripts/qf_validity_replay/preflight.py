"""Read-only pinned-handoff entry point; never fetch or move a ref implicitly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


EXPECTED_REF = "refs/heads/work/qf-august-data-review-20261001"
DEFAULT_HANDOFF = "docs/qf_data_review/20261001/handoff.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--producer-sha", help="Exact inspected full producer SHA, never a branch name")
    parser.add_argument("--handoff-path", default=DEFAULT_HANDOFF)
    args = parser.parse_args()
    if not args.producer_sha:
        print(json.dumps({
            "status": "WAITING_FOR_DATA_HANDOFF", "required_ref": EXPECTED_REF,
            "required_path": args.handoff_path, "producer_sha": None,
            "next": "Resolve the advertised exact task ref, fetch/filter and inspect it, then pin its full SHA.",
            "scoring_run": False, "training_or_inference_run": False,
        }, indent=2))
        return 3
    if not re.fullmatch(r"[0-9a-f]{40}", args.producer_sha):
        parser.error("--producer-sha must be an exact lowercase full SHA-1 commit ID")
    relative = Path(args.handoff_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        parser.error("handoff path must be repository-relative without parent traversal")
    env = os.environ.copy()
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_LFS_SKIP_SMUDGE="1", GIT_NO_LAZY_FETCH="1")
    def git(*argv):
        return subprocess.run(["git", "-C", str(args.repository), *argv],
                              capture_output=True, env=env, check=True)
    # Enumerate locally stored objects first. This also prevents implicit
    # promisor retrieval on Git versions that ignore GIT_NO_LAZY_FETCH.
    object_lines = git("cat-file", "--batch-all-objects",
                       "--batch-check=%(objectname) %(objecttype)").stdout.decode().splitlines()
    local_objects = dict(line.split() for line in object_lines)
    resolved = args.producer_sha
    if local_objects.get(resolved) != "commit":
        raise ValueError("Pinned producer commit is not already local; fetch it explicitly first")
    commit = git("cat-file", "-p", resolved).stdout.decode()
    tree = next(line.split()[1] for line in commit.splitlines() if line.startswith("tree "))
    for position, component in enumerate(relative.parts):
        if local_objects.get(tree) != "tree":
            raise ValueError("Required tree is not already local; retrieve it explicitly first")
        entries = git("ls-tree", "-z", tree).stdout.split(b"\0")
        selected = []
        for entry in entries:
            if entry:
                metadata, name = entry.split(b"\t", 1)
                if name.decode() == component:
                    selected.append(metadata.decode().split())
        if len(selected) != 1:
            raise ValueError("Pinned handoff path does not exist")
        _, kind, oid = selected[0]
        if position < len(relative.parts) - 1:
            if kind != "tree":
                raise ValueError("Handoff parent is not a directory")
            tree = oid
        elif kind != "blob" or local_objects.get(oid) != "blob":
            raise ValueError("Handoff blob is not already local; retrieve it explicitly first")
    payload = git("cat-file", "blob", oid).stdout
    handoff = json.loads(payload)
    if not isinstance(handoff, dict):
        raise ValueError("Expected a JSON object handoff")
    print(json.dumps({
        "status": "PINNED_HANDOFF_REQUIRES_SCHEMA_AND_MANIFEST_REVIEW",
        "producer_sha": resolved, "handoff_path": args.handoff_path,
        "handoff_sha256": hashlib.sha256(payload).hexdigest(),
        "handoff_fields": sorted(handoff),
        "next": "Verify producer lineage/checksums/serialization and frozen artifact index before adapting rows to replay_core.py.",
        "scoring_run": False, "training_or_inference_run": False,
    }, indent=2))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
