import os, json, time, random, yaml, pandas as pd, numpy as np, subprocess
from pyswarms.single.global_best import GlobalBestPSO

VERSION    = os.environ.get("YOLO_VERSION","YOLOv8")
DATA_YAML  = os.environ.get("DATA_YAML","/content/SeminarYOLO/gwhd.yaml")
RUN_ID     = os.environ.get("RUN_ID","run_01")
N_PART     = int(os.environ.get("N_PARTICLES","3"))
N_ITERS    = int(os.environ.get("N_ITERS","2"))
EPOCHS_FIT = int(os.environ.get("EPOCHS_FIT","2"))

SAVE_DIR = f"/content/SeminarYOLO/results/{VERSION}/LFPSO/{RUN_ID}"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("/content/SeminarYOLO/configs/search_space.yaml") as f:
    S = yaml.safe_load(f)

keys = ["lr0","lrf","momentum","weight_decay","iou","conf",
        "hsv_h","hsv_s","hsv_v","degrees","translate","scale","mosaic","mixup","imgsz"]
lb = np.array([S[k][0] for k in keys], dtype=float)
ub = np.array([S[k][1] for k in keys], dtype=float)

def _round_discrete(params):
    params[-1] = 640 if params[-1] < 768 else 896
    return params

def levy_step(beta=1.5, size=None, scale=0.02):
    from math import gamma, pi
    if size is None: size = 1
    sigma_u = (gamma(1+beta)*np.sin(pi*beta/2) /
              (gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma_u, size)
    v = np.random.normal(0, 1, size)
    return scale * (u / (np.abs(v)**(1/beta)))

def _train_eval(params, tag):
    p = dict(zip(keys, params))
    p = dict(zip(keys, _round_discrete(list(p.values()))))
    name = f"lfpso_fit_{tag}_{random.randint(0,1_000_000)}"
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

# logging helpers
_iter = {"i": 0}
def _save_iter_csv(save_dir, iter_idx, rows):
    df = pd.DataFrame(rows)
    one = os.path.join(save_dir, f"iter_{iter_idx:02d}.csv")
    df.to_csv(one, index=False)
    cum = os.path.join(save_dir, "iter_logs.csv")
    if os.path.exists(cum):
        pd.concat([pd.read_csv(cum), df], ignore_index=True).to_csv(cum, index=False)
    else:
        df.to_csv(cum, index=False)

# objective with Lévy perturbation
def make_objective(optimizer):
    def objective(X):
        it = _iter["i"]
        # Lévy perturb every 3 iterations (after the first)
        if it > 0 and it % 3 == 0:
            X[:] = np.clip(X + levy_step(size=X.shape, scale=0.02), lb, ub)
        rows, costs = [], []
        for i, x in enumerate(X):
            tag = f"it{it}_p{i}"
            m, exp_name = _train_eval(x.copy(), tag)
            row = dict(iter=it, particle=i, model=VERSION, run=RUN_ID,
                       val_mAP50_95=m, wall_time=time.time(), exp_name=exp_name)
            for k, val in zip(keys, x):
                row[k] = float(val)
            rows.append(row)
            costs.append(-m)
        _save_iter_csv(SAVE_DIR, it, rows)
        _iter["i"] = it + 1
        return np.array(costs)
    return objective

optimizer = GlobalBestPSO(n_particles=N_PART,
                          dimensions=len(keys),
                          options={'c1':1.5,'c2':1.5,'w':0.6},
                          bounds=(lb, ub))

best_cost, best_pos = optimizer.optimize(make_objective(optimizer),
                                         iters=N_ITERS, verbose=True)

best = {"best_cost": float(best_cost),
        "best_params": dict(zip(keys, _round_discrete(list(best_pos.copy()))))}
with open(os.path.join(SAVE_DIR, "best.json"), "w") as f:
    json.dump(best, f, indent=2)
print("BEST_mAP50_95 (approx)", -best_cost)
