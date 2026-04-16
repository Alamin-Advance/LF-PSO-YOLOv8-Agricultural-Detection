import os, json, time, random, yaml, pandas as pd, numpy as np, subprocess
from pyswarms.single.global_best import GlobalBestPSO

# ------------ Config from environment ------------
VERSION    = os.environ.get("YOLO_VERSION","YOLOv8")
DATA_YAML  = os.environ.get("DATA_YAML","/content/SeminarYOLO/gwhd.yaml")
RUN_ID     = os.environ.get("RUN_ID","run_01")
N_PART     = int(os.environ.get("N_PARTICLES","3"))
N_ITERS    = int(os.environ.get("N_ITERS","2"))
EPOCHS_FIT = int(os.environ.get("EPOCHS_FIT","2"))

SAVE_DIR = f"/content/SeminarYOLO/results/{VERSION}/PSO/{RUN_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("/content/SeminarYOLO/configs/search_space.yaml") as f:
    S = yaml.safe_load(f)

# ------------ Search space ------------
keys = ["lr0","lrf","momentum","weight_decay","iou","conf",
        "hsv_h","hsv_s","hsv_v","degrees","translate","scale","mosaic","mixup","imgsz"]
lb = np.array([S[k][0] for k in keys], dtype=float)
ub = np.array([S[k][1] for k in keys], dtype=float)

def _round_discrete(params):
    # imgsz: {640, 896}
    params[-1] = 640 if params[-1] < 768 else 896
    return params

def _train_eval(params, tag):
    """Train a short model with given params; return validation mAP50-95."""
    p = dict(zip(keys, params))
    p = dict(zip(keys, _round_discrete(list(p.values()))))
    name = f"pso_fit_{tag}_{random.randint(0,1_000_000)}"
    train_cmd = [
      "yolo","detect","train","model=yolov8n.pt",f"data={DATA_YAML}",
      f"imgsz={int(p['imgsz'])}",f"epochs={EPOCHS_FIT}","batch=16","device=0",
      *(f"{k}={p[k]}" for k in ["lr0","lrf","momentum","weight_decay","iou","conf",
                                "hsv_h","hsv_s","hsv_v","degrees","translate","scale",
                                "mosaic","mixup"]),
      f"project=/content/SeminarYOLO/results/{VERSION}","name="+name
    ]
    subprocess.run(train_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    v = subprocess.run([
      "yolo","detect","val",
      f"model=/content/SeminarYOLO/results/{VERSION}/{name}/weights/best.pt",f"data={DATA_YAML}",
      "save_json=False"
    ], capture_output=True, text=True, check=True)
    lines = [l for l in v.stdout.splitlines() if "mAP50-95" in l]
    m = float(lines[-1].split()[-1]) if lines else 0.0
    return m, name

# iteration counter closure so we can write one CSV per iteration
_iter = {"i": 0}

def _save_iter_csv(iter_idx, rows):
    df = pd.DataFrame(rows)
    one = os.path.join(SAVE_DIR, f"iter_{iter_idx:02d}.csv")
    df.to_csv(one, index=False)
    cum = os.path.join(SAVE_DIR, "iter_logs.csv")
    if os.path.exists(cum):
        pd.concat([pd.read_csv(cum), df], ignore_index=True).to_csv(cum, index=False)
    else:
        df.to_csv(cum, index=False)

# ------------ Objective for pyswarms ------------
def objective(X):
    """
    X: (n_particles, n_dims)
    Return: array of costs (we minimize negative mAP50-95)
    Also logs a CSV for this iteration.
    """
    it = _iter["i"]
    rows, costs = [], []
    for i, x in enumerate(X):
        tag = f"it{it}_p{i}"
        m, exp_name = _train_eval(x.copy(), tag)
        row = dict(iter=it, particle=i, model=VERSION, run=RUN_ID,
                   val_mAP50_95=m, wall_time=time.time(), exp_name=exp_name)
        for k, val in zip(keys, x):
            row[k] = float(val)
        rows.append(row)
        costs.append(-m)  # minimize
    _save_iter_csv(it, rows)
    _iter["i"] = it + 1
    return np.array(costs)

# ------------ Run PSO ------------
optimizer = GlobalBestPSO(n_particles=N_PART,
                          dimensions=len(keys),
                          options={'c1':1.5,'c2':1.5,'w':0.6},
                          bounds=(lb, ub))

best_cost, best_pos = optimizer.optimize(objective, iters=N_ITERS, verbose=True)

best = {
  "best_cost": float(best_cost),
  "best_params": dict(zip(keys, _round_discrete(list(best_pos.copy()))))
}
with open(os.path.join(SAVE_DIR, "best.json"), "w") as f:
    json.dump(best, f, indent=2)

print("BEST_mAP50_95 (approx)", -best_cost)
