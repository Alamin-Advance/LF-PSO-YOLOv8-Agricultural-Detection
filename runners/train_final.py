import json, subprocess, sys, os

VERSION, RUN_ID, DATA_YAML = sys.argv[1], sys.argv[2], sys.argv[3]
best_path = f"/content/SeminarYOLO/results/{VERSION}/PSO/{RUN_ID}/best.json"
with open(best_path) as f:
    best = json.load(f)
p = best["best_params"]

name = f"final_PSO_{RUN_ID}"
cmd = [
 "yolo","detect","train","model=yolov8n.pt",f"data={DATA_YAML}",
 f"imgsz={int(p['imgsz'])}","epochs=2","batch=4","device=0",
 *(f"{k}={v}" for k,v in p.items() if k!="imgsz"),
 f"project=/content/SeminarYOLO/results/{VERSION}", f"name={name}"
]
subprocess.run(cmd, check=True)
subprocess.run([
 "yolo","detect","val",
 f"model=/content/SeminarYOLO/results/{VERSION}/{name}/weights/best.pt",
 f"data={DATA_YAML}"
], check=True)
print("Final model:", f"/content/SeminarYOLO/results/{VERSION}/{name}/weights/best.pt")
