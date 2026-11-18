"""
AI Stock Image Pipeline - launch-and-terminate smoke test
-------------------------------------------------
What this does:
  1. Tries to launch a ComfyUI pod, working through a list of cheap GPUs
     (Community Cloud first, then Secure Cloud).
  2. Waits until the pod is RUNNING and ComfyUI answers on port 8188.
  3. Terminates the pod - ALWAYS, even if something fails or you press Ctrl+C.

Expected cost: a few US cents (pod runs ~3-6 minutes).

Run it from the pipeline folder with the .venv active:
    python launch_test.py
"""

import os
import sys
import time

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------- settings
API = "https://api.runpod.io/v2"
IMAGE = "runpod/comfyui:2.2.1-comfyuiv0.44.0-cuda13.2"   # the one that worked in Step 2
MIN_CUDA = "13.2"          # skip machines whose driver can't run this image
DISK_GB = 50               # container disk (wiped on terminate - that's fine)

# GPUs to try, cheapest/most suitable first. Any 16GB+ card runs SDXL.
GPU_PREFERENCES = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA L4",
    "NVIDIA A40",
]
CLOUDS = ["COMMUNITY", "SECURE"]   # cheaper first

STARTUP_TIMEOUT_MIN = 15   # give up waiting for RUNNING/ComfyUI after this
# --------------------------------------------------------------------------

load_dotenv()
KEY = os.getenv("RUNPOD_API_KEY")
if not KEY:
    sys.exit("RUNPOD_API_KEY not found - check your .env file.")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def create_pod():
    """Try each cloud + GPU combination until one launches. Returns (pod_id, gpu, cloud)."""
    for cloud in CLOUDS:
        for gpu in GPU_PREFERENCES:
            body = {
                "name": "lumen-lane-test",
                "cloud": cloud,
                "image": IMAGE,
                "disk": DISK_GB,
                "ports": ["8188/http"],
                "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA},
            }
            r = requests.post(f"{API}/pods", headers=HEADERS, json=body, timeout=60)
            if r.status_code == 201:
                pod = r.json()
                print(f"  ✅ Launched on {gpu} ({cloud}) - pod id {pod['id']}")
                return pod["id"], gpu, cloud
            # Not available / not valid -> show a short reason and try the next one
            reason = r.text[:120].replace("\n", " ")
            print(f"  ✖ {gpu} ({cloud}): {r.status_code} {reason}")
    return None, None, None


def get_pod(pod_id):
    r = requests.get(f"{API}/pods/{pod_id}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def terminate(pod_id):
    """Delete the pod. Retries a few times because this is the step that saves money."""
    for attempt in range(1, 6):
        try:
            r = requests.delete(f"{API}/pods/{pod_id}", headers=HEADERS, timeout=30)
            if r.status_code in (200, 202, 204, 404):
                print(f"  🛑 Pod {pod_id} terminated (HTTP {r.status_code}). Billing stopped.")
                return True
            print(f"  ⚠ Terminate attempt {attempt} returned {r.status_code}: {r.text[:120]}")
        except requests.RequestException as e:
            print(f"  ⚠ Terminate attempt {attempt} failed: {e}")
        time.sleep(5)
    print("  ‼ COULD NOT TERMINATE AUTOMATICALLY - go to console.runpod.io → Pods and terminate it by hand!")
    return False


def wait_until_ready(pod_id):
    """Wait for status RUNNING, then for ComfyUI to answer through the RunPod proxy."""
    deadline = time.time() + STARTUP_TIMEOUT_MIN * 60
    status = None
    while time.time() < deadline:
        status = get_pod(pod_id).get("status")
        print(f"  … pod status: {status}")
        if status == "RUNNING":
            break
        if status in ("ERROR", "EXITED", "TERMINATED"):
            print("  ✖ Pod failed to start.")
            return False
        time.sleep(15)
    else:
        print("  ✖ Timed out waiting for RUNNING.")
        return False

    url = f"https://{pod_id}-8188.proxy.runpod.net/system_stats"
    print(f"  … waiting for ComfyUI at {url}")
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=15)
            if r.ok:
                stats = r.json()
                devices = stats.get("devices", [])
                gpu_name = devices[0].get("name", "?") if devices else "?"
                print(f"  ✅ ComfyUI is up! GPU reported by ComfyUI: {gpu_name}")
                return True
        except requests.RequestException:
            pass
        time.sleep(15)
    print("  ✖ Timed out waiting for ComfyUI to respond.")
    return False


def main():
    print("Step 1/3: launching a pod...")
    pod_id, gpu, cloud = create_pod()
    if not pod_id:
        print("No GPU available right now on any option. Nothing was charged. Try again later.")
        return

    started = time.time()
    try:
        print("Step 2/3: waiting for ComfyUI to be ready...")
        ok = wait_until_ready(pod_id)
        print("RESULT:", "SUCCESS 🎉" if ok else "did not become ready (see messages above)")
    except KeyboardInterrupt:
        print("\n  Ctrl+C pressed - cleaning up.")
    finally:
        print("Step 3/3: terminating the pod...")
        terminate(pod_id)
        mins = (time.time() - started) / 60
        print(f"Pod was alive for about {mins:.1f} minutes on {gpu} ({cloud}).")


if __name__ == "__main__":
    main()
