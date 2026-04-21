"""
push_to_kaggle.py
=================
Envia o projeto SCAO como dataset + kernel para o Kaggle usando a API REST diretamente.
Não depende do kaggle CLI (incompatível com Python 3.14).

Uso:
    python scripts/push_to_kaggle.py

Requer:
    C:\\Users\\<user>\\.kaggle\\kaggle.json  com  {"username":"...","key":"..."}
"""

import json, os, sys, zipfile, requests, pathlib, time
from base64 import b64encode

# ── Load credentials ──────────────────────────────────────────────────────────
CRED_PATH = pathlib.Path.home() / ".kaggle" / "kaggle.json"
if not CRED_PATH.exists():
    print(f"ERROR: {CRED_PATH} not found.")
    print("Go to kaggle.com → Settings → API → Create New Token and place the file there.")
    sys.exit(1)

creds = json.loads(CRED_PATH.read_text())
USERNAME = creds["username"]
API_KEY   = creds["key"]
AUTH = (USERNAME, API_KEY)
BASE = "https://www.kaggle.com/api/v1"

print(f"Authenticated as: {USERNAME}")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATASET_SLUG = "scao-project"
KERNEL_SLUG  = "scao-v011-gpu-benchmark"

# ── Step 1: Zip the project ───────────────────────────────────────────────────
ZIP_PATH = PROJECT_ROOT / "scao_project_upload.zip"

EXCLUDE_PATTERNS = {
    "__pycache__", ".git", ".mypy_cache", ".ruff_cache",
    "scao.egg-info", "dist", "build", "*.pyc", "*.pyo",
    "scao_project_upload.zip",
    # exclude large result files
    "results_v", "results_multiscale", "results_benchmark",
    "results_wikitext", "results_scao",
}

def should_exclude(path_str: str) -> bool:
    parts = pathlib.Path(path_str).parts
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return True
        for pat in EXCLUDE_PATTERNS:
            if pat.startswith("*.") and part.endswith(pat[1:]):
                return True
        if "results_v" in part or "results_multiscale" in part:
            return True
    return False

print(f"\nCreating zip: {ZIP_PATH.name} ...")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in PROJECT_ROOT.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(PROJECT_ROOT))
            if not should_exclude(rel):
                zf.write(f, rel)

size_mb = ZIP_PATH.stat().st_size / 1e6
print(f"Zip created: {size_mb:.1f} MB")

# ── Step 2: Create or update the Dataset ─────────────────────────────────────
# Check if dataset exists
r = requests.get(f"{BASE}/datasets/{USERNAME}/{DATASET_SLUG}", auth=AUTH)

if r.status_code == 200:
    print(f"\nDataset '{USERNAME}/{DATASET_SLUG}' exists — creating new version...")
    # Upload file blob first
    blob_r = requests.post(
        f"{BASE}/blobs/upload",
        auth=AUTH,
        json={
            "name": "scao_project.zip",
            "lastModifiedEpochSeconds": int(time.time()),
        }
    )
    blob_r.raise_for_status()
    blob_info = blob_r.json()
    token = blob_info["token"]
    upload_url = blob_info["createUrl"]

    # Upload the actual file to the blob URL
    print("Uploading zip...")
    with open(ZIP_PATH, "rb") as fh:
        up_r = requests.put(upload_url, data=fh,
                            headers={"Content-Type": "application/zip"})
    up_r.raise_for_status()
    print("Upload complete.")

    # Create new dataset version
    ver_r = requests.post(
        f"{BASE}/datasets/{USERNAME}/{DATASET_SLUG}/versions",
        auth=AUTH,
        json={
            "versionNotes": "SCAO v0.1.1 — int8 EMA + CUDA kernels",
            "files": [{"token": token}],
        }
    )
    ver_r.raise_for_status()
    print(f"Dataset version created: {ver_r.json()}")

else:
    print(f"\nDataset '{USERNAME}/{DATASET_SLUG}' not found — creating new dataset...")
    # Create dataset
    blob_r = requests.post(
        f"{BASE}/blobs/upload",
        auth=AUTH,
        json={
            "name": "scao_project.zip",
            "lastModifiedEpochSeconds": int(time.time()),
        }
    )
    blob_r.raise_for_status()
    blob_info = blob_r.json()
    token = blob_info["token"]
    upload_url = blob_info["createUrl"]

    print("Uploading zip...")
    with open(ZIP_PATH, "rb") as fh:
        up_r = requests.put(upload_url, data=fh,
                            headers={"Content-Type": "application/zip"})
    up_r.raise_for_status()
    print("Upload complete.")

    create_r = requests.post(
        f"{BASE}/datasets",
        auth=AUTH,
        json={
            "ownerSlug": USERNAME,
            "slug": DATASET_SLUG,
            "title": "SCAO Project v0.1.1",
            "isPrivate": True,
            "files": [{"token": token}],
            "convertToCsv": False,
        }
    )
    create_r.raise_for_status()
    print(f"Dataset created: https://www.kaggle.com/datasets/{USERNAME}/{DATASET_SLUG}")

# ── Step 3: Push the Kernel ───────────────────────────────────────────────────
kernel_script = (PROJECT_ROOT / "scripts" / "kaggle_kernel_scao.py").read_text(encoding="utf-8")

# Replace placeholder username in metadata
kernel_body = {
    "source": kernel_script,
    "language": "python",
    "kernelType": "script",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [f"{USERNAME}/{DATASET_SLUG}"],
    "kernelDataSources": [],
    "competitionDataSources": [],
}

# Check if kernel exists
k_check = requests.get(f"{BASE}/kernels/{USERNAME}/{KERNEL_SLUG}", auth=AUTH)

if k_check.status_code == 200:
    print(f"\nKernel '{KERNEL_SLUG}' exists — pushing new version...")
    push_r = requests.post(
        f"{BASE}/kernels/push",
        auth=AUTH,
        json={
            "id": f"{USERNAME}/{KERNEL_SLUG}",
            "title": "SCAO v0.1.1 — GPU Benchmark",
            "newTitle": "SCAO v0.1.1 — GPU Benchmark",
            **kernel_body,
        }
    )
else:
    print(f"\nCreating new kernel '{KERNEL_SLUG}'...")
    push_r = requests.post(
        f"{BASE}/kernels/push",
        auth=AUTH,
        json={
            "id": f"{USERNAME}/{KERNEL_SLUG}",
            "title": "SCAO v0.1.1 — GPU Benchmark",
            **kernel_body,
        }
    )

if push_r.status_code in (200, 201):
    info = push_r.json()
    print(f"Kernel pushed successfully!")
    print(f"URL: https://www.kaggle.com/code/{USERNAME}/{KERNEL_SLUG}")
    print(f"Version: {info.get('versionNumber', '?')}")
    print(f"\nThe kernel will start running automatically on a T4 GPU.")
    print(f"Check status at: https://www.kaggle.com/code/{USERNAME}/{KERNEL_SLUG}")
else:
    print(f"ERROR pushing kernel: {push_r.status_code} {push_r.text}")

# ── Cleanup ───────────────────────────────────────────────────────────────────
ZIP_PATH.unlink(missing_ok=True)
print("\nDone. Zip cleaned up.")
