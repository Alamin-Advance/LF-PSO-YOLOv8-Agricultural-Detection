import os, json, subprocess, sys

if len(sys.argv) != 5:
    print("Usage: python train_final_generic.py <YOLO_VERSION> <OPTIMIZER> <RUN_ID> <DATA_YAML>")
    sys.exit(1)

VERSION, OPTIM, RUN_ID, DATA_YAML = sys.argv[1:]
RUN_DIR = f"/content/SeminarYOLO/results/{VERSION}/{OPTIM}/{RUN_ID}"
BEST_JSON = os.path.join(RUN_DIR, "best.json")

if not os.path.exists(BEST_JSON):
    sys.exit(f" best.json not found at {BEST_JSON}")

with open(BEST_JSON) as f:
    best = json.load(f)

params = best["best_params"]
out_dir = f"/content/SeminarYOLO/results/{VERSION}/final_{OPTIM}_{RUN_ID}"
os.makedirs(out_dir, exist_ok=True)

print(f" Using best params from {BEST_JSON}")
print(json.dumps(params, indent=2))

train_cmd = [
    "yolo", "detect", "train",
    "model=yolov8n.pt",
    f"data={DATA_YAML}",
    "epochs=2", "batch=4", "device=0",
    f"project=/content/SeminarYOLO/results/{VERSION}",
    f"name=final_{OPTIM}_{RUN_ID}"
]
for k, v in params.items():
    train_cmd.append(f"{k}={v}")

subprocess.run(train_cmd, check=True)
print(f"✅ Training complete. Results saved in {out_dir}")
