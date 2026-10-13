"""Phase 2 training: visual-only and visual+category, seeds 0-4, SAME config as selected in Phase 1 (chosen on val_id only; not re-tuned per variant).
Early stopping on val_id NLL. After training, logits for val_id / test_id / test_ud / 2014 / 2015 / 2017 are stored (no model selection uses them).
Run: PYTHONPATH=./pylibs:./scripts python3 -u scripts/52_phase2_train.py 2>&1 | tee results/exp1_phase2_train_log.txt
"""
import json, time
import numpy as np
from taskb_lib import *

OUT = os.path.join(ROOT, "runs", "phase2"); os.makedirs(OUT, exist_ok=True)
sel = json.load(open(os.path.join(ROOT, "runs", "phase1", "selected_config.json")))["selected"]
print("config from Phase 1 (val_id NLL):", sel)
data = Data()
roles = ["val_id", "test_id", "test_ud", "test_2014", "test_2015", "test_2017"]
EV = {r: data.load_eval(r) for r in roles}
tab = np.load(os.path.join(RES, "exp1_instance_table_taskB.npz"))
for r in roles:
    assert list(tab[f"{r}/instance_id"]) == EV[r]["ids"], f"instance order mismatch {r}"
probe_s = data.sample_train(np.random.default_rng(777), idx=np.random.default_rng(778).choice(data.N, 3851, replace=False))
probe = dict(ctx=probe_s["ctx"], cands=probe_s["cands"], label=probe_s["label"])
summary = {}
for variant in ("visual", "visual_cat"):
    for seed in range(5):
        cfg = dict(sel, variant=variant)
        d = os.path.join(OUT, f"{variant}_seed{seed}")
        t0 = time.time()
        model, info, log = train_run(data, cfg, seed, EV["val_id"], probe, d, verbose=False)
        preds = {r: predict(model, data, EV[r]) for r in roles}
        np.savez(os.path.join(d, "preds.npz"), **preds)
        json.dump(dict(cmd="PYTHONPATH=./pylibs:./scripts python3 scripts/52_phase2_train.py", env=env_info()), open(os.path.join(d, "env.json"), "w"), indent=1)
        accs = {r: metrics(preds[r], EV[r]["label"])["acc"] for r in roles}
        summary[f"{variant}_seed{seed}"] = dict(info=info, acc=accs)
        print(f"{variant} seed {seed}: best_epoch={info['best_epoch']} params={info['params']:,} train_s={info['train_seconds']} | " +
              " ".join(f"{r}={accs[r]*100:.1f}" for r in roles), flush=True)
json.dump(summary, open(os.path.join(OUT, "phase2_summary.json"), "w"), indent=1)
