"""Partitioned aggregate for datasets too big to fit one (split,Re) tensor in RAM.
Writes train_Re{re}_p{i}.pt (PART shards each) instead of one train_Re{re}.pt, deleting each
shard group after it's written so peak disk stays ~constant. The datamodule loads
train_Re{re}_p*.pt as a ConcatDataset (train_simple). Val aggregated normally.
    python aggregate_partitioned.py DATASET_ROOT [SHARDS_PER_PARTITION]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from aggregate_dataset import aggregate
def main(root, part):
    for sd in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(sd.glob("*.npy"))
        if not paths: continue
        metas = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
        sdt = metas[0]["resolved"]["snapshot_dt"]
        by_re = defaultdict(list)
        for p,m in zip(paths,metas): by_re[int(m["config"]["Re"])].append((p,m))
        for re_val, items in sorted(by_re.items()):
            if sd.name == "train" and part > 0 and len(items) > part:
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
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv)>2 else 8)
