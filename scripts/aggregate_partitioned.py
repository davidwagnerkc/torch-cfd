"""Partitioned aggregate for datasets too big to fit one (split,Re) tensor in RAM.
Writes <split>_Re{re}_p{i}.pt (PART shards each) instead of one <split>_Re{re}.pt, deleting each
shard group after it's written so peak disk stays ~constant. The datamodule loads
train_Re{re}_p*.pt as a ConcatDataset (train_simple). Val aggregated normally.

PARTITIONED_SPLITS is which splits get the partitioned treatment. `test` is in it for the official
1024-trajectory Re1000 test set (2026-10-01): 64 shards of 16 -> 8 x test_Re1000_p{i}.pt of 128
trajectories, each structurally identical to val_Re1000.pt so every existing loader and eval works
on a partition unchanged. Partitioning is also what keeps peak disk sane -- the full set is ~96 GB.

    python aggregate_partitioned.py DATASET_ROOT [SHARDS_PER_PARTITION] [SPLIT]

SPLIT (optional) restricts the walk to one split dir, so aggregating a new split cannot touch the
shards of a split that is already aggregated.

Shards are ordered by NUMERIC rank, not lexicographically. Shard files are "{rank}-Re...", so a
plain sort gives 0,1,10,11,..,19,2,20,.. and partition p0 would hold ranks 0,1,10..15 -- every
shard still lands in exactly one partition, but p{i}'s seed range is scrambled, which makes the
released per-partition provenance unreadable. Numeric order gives p{i} = ranks part*i..part*i+part-1
= one contiguous seed block. (Pre-existing train_*_p*.pt were built under the old order; their
union is unchanged, only the grouping would differ on a regenerate.)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from aggregate_dataset import aggregate
PARTITIONED_SPLITS = ("train", "test")
def _rank(p):
    """Sort key: numeric shard rank ("12-Re1000-ds-..." -> 12), name for anything else."""
    head = p.name.split("-")[0]
    return (0, int(head), "") if head.isdigit() else (1, 0, p.name)
def main(root, part, only=None):
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and sd.name != only: continue
        paths = sorted(sd.glob("*.npy"), key=_rank)
        if not paths: continue
        metas = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
        sdt = metas[0]["resolved"]["snapshot_dt"]
        by_re = defaultdict(list)
        for p,m in zip(paths,metas): by_re[int(m["config"]["Re"])].append((p,m))
        for re_val, items in sorted(by_re.items()):
            if sd.name in PARTITIONED_SPLITS and part > 0 and len(items) > part:
                for pi in range(0, len(items), part):
                    out = sd/f"{sd.name}_Re{re_val}_p{pi//part}.pt"
                    if out.exists(): print("skip", out.name, flush=True); continue
                    grp = items[pi:pi+part]; n = aggregate(grp, sdt, out)
                    for p,_ in grp: p.unlink(missing_ok=True); p.with_suffix(".json").unlink(missing_ok=True)
                    print(f"{out.name}: N={n} (-{len(grp)} shards)", flush=True)
            else:
                out = sd/f"{sd.name}_Re{re_val}.pt"
                if out.exists(): print("skip", out.name, flush=True); continue
                print(f"{out.name}: N={aggregate(items, sdt, out)}", flush=True)
    print("PARTITION AGGREGATE DONE", flush=True)
if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 8,
         sys.argv[3] if len(sys.argv)>3 else None)
