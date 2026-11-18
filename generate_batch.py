"""
Lumen & Lane - batch generator (v3: Z-Image Turbo)
--------------------------------------------------
What this does (no clicking needed):
  1. Launches a 24GB+ GPU pod (best available from the list, Community first).
  2. Connects over SSH and downloads the model files onto the pod:
       - Z-Image Turbo (Apache 2.0 - commercial use OK) + its text encoder + VAE
       - RealESRGAN x2plus upscaler (BSD-3 - commercial use OK)
  3. For every row in prompts.csv, generates IMAGES_PER_PROMPT variations at
     1920x1088, then upscales 2x on the pod -> 3840x2176 (~8.4 MP).
  4. Downloads each image to output/<date-time>/ on your laptop as a high quality
     JPEG with an Adobe-safe filename, plus:
       - adobe_upload_<tag>.csv (Filename, Title, Keywords, Category, Releases)
       - log.csv               (prompt and seed used for every image)
  5. Terminates the pod - ALWAYS, even on errors, Ctrl+C or the time limit.

Run from the pipeline folder with (.venv) active:
    python generate_batch.py
"""

import csv
import io
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import paramiko
import requests
from dotenv import load_dotenv
from PIL import Image

# ============================================================== SETTINGS
IMAGES_PER_PROMPT = 3          # variations per prompt - you keep the best 1-2
WIDTH, HEIGHT = 1920, 1088     # native generation size (multiples of 16)
STEPS = 8                      # Z-Image Turbo is tuned for ~8 steps
MAX_RUNTIME_MIN = 120          # hard cap: pod is terminated after this, no matter what
MAX_CONSECUTIVE_FAILURES = 3   # stop early (and terminate) if something is clearly broken
COMFY_START_TIMEOUT_MIN = 8    # if ComfyUI isn't up by then, the pod is faulty -> replace it
MAX_POD_ATTEMPTS = 3           # how many faulty pods to replace before giving up

PROMPTS_FILE = "prompts.csv"   # columns: theme,prompt,title,keywords,category
OUTPUT_ROOT = "output"
SSH_KEY = Path.home() / ".ssh" / "runpod_lumen"
JPEG_QUALITY = 95
FILE_PREFIX = "LL"             # Adobe filenames must be 30 characters or fewer

API = "https://api.runpod.io/v2"
IMAGE = "runpod/comfyui:2.2.1-comfyuiv0.44.0-cuda13.2"
MIN_CUDA = "13.2"
DISK_GB = 60
# 24GB+ cards only - Z-Image + its text encoder need the headroom. Fastest first.
GPU_PREFERENCES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
]
CLOUDS = ["SECURE", "COMMUNITY"]   # Secure first: Community hosts kept failing GPU init

HF = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"
# (subfolder under ComfyUI/models, filename, url, minimum expected size in bytes)
MODEL_FILES = [
    ("diffusion_models", "z_image_turbo_bf16.safetensors", f"{HF}/diffusion_models/z_image_turbo_bf16.safetensors", 10_000_000_000),
    ("text_encoders", "qwen_3_4b.safetensors", f"{HF}/text_encoders/qwen_3_4b.safetensors", 6_000_000_000),
    ("vae", "ae.safetensors", f"{HF}/vae/ae.safetensors", 200_000_000),
    ("upscale_models", "RealESRGAN_x2plus.pth",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 10_000_000),
]
# =======================================================================

load_dotenv()
KEY = os.getenv("RUNPOD_API_KEY")
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
START = time.time()


def log(msg):
    print(f"[{(time.time() - START) / 60:5.1f} min] {msg}", flush=True)


def time_left():
    return MAX_RUNTIME_MIN * 60 - (time.time() - START)


# ----------------------------------------------------------- pre-flight checks
def preflight():
    problems = []
    if not KEY:
        problems.append("RUNPOD_API_KEY missing from .env")
    if not SSH_KEY.exists() or not SSH_KEY.with_suffix(".pub").exists():
        problems.append(f"SSH key not found at {SSH_KEY} (+ .pub)")
    if not Path(PROMPTS_FILE).exists():
        problems.append(f"{PROMPTS_FILE} not found in this folder")
    if problems:
        sys.exit("Can't start:\n  - " + "\n  - ".join(problems))

    with open(PROMPTS_FILE, newline="", encoding="utf-8-sig") as f:
        prompts = [row for row in csv.DictReader(f) if (row.get("prompt") or "").strip()]
    for row in prompts:
        missing = [c for c in ("prompt", "title", "keywords", "category") if not (row.get(c) or "").strip()]
        if missing:
            sys.exit(f"{PROMPTS_FILE}: a row is missing {missing}: {row.get('prompt', '')[:60]}")
    if not prompts:
        sys.exit(f"{PROMPTS_FILE} has no prompts in it.")
    return prompts


# ----------------------------------------------------------------- RunPod API
def create_pod(excluded=()):
    pub_key = SSH_KEY.with_suffix(".pub").read_text().strip()
    for cloud in CLOUDS:
        for gpu in GPU_PREFERENCES:
            if gpu in excluded:
                continue
            body = {
                "name": "lumen-lane-batch",
                "cloud": cloud,
                "image": IMAGE,
                "disk": DISK_GB,
                "ports": ["8188/http", "22/tcp"],
                "env": {"PUBLIC_KEY": pub_key},
                "startSsh": True,
                "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA},
            }
            r = requests.post(f"{API}/pods", headers=HEADERS, json=body, timeout=60)
            if r.status_code == 201:
                pod_id = r.json()["id"]
                log(f"✅ Launched {gpu} ({cloud}) - pod {pod_id}")
                return pod_id, gpu, cloud
            log(f"✖ {gpu} ({cloud}): {r.status_code} {r.text[:90]}")
    return None, None, None


def get_pod(pod_id):
    r = requests.get(f"{API}/pods/{pod_id}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def terminate(pod_id):
    for attempt in range(1, 6):
        try:
            r = requests.delete(f"{API}/pods/{pod_id}", headers=HEADERS, timeout=30)
            if r.status_code in (200, 202, 204, 404):
                log(f"🛑 Pod {pod_id} terminated (HTTP {r.status_code}). Billing stopped.")
                return
            log(f"⚠ Terminate attempt {attempt}: {r.status_code} {r.text[:100]}")
        except requests.RequestException as e:
            log(f"⚠ Terminate attempt {attempt} failed: {e}")
        time.sleep(5)
    log("‼ COULD NOT TERMINATE - go to console.runpod.io → Pods and terminate it by hand!")


def wait_for_ssh_port(pod_id, timeout=600):
    deadline = time.time() + timeout
    while time_left() > 0 and time.time() < deadline:
        pod = get_pod(pod_id)
        status = pod.get("status")
        if status in ("ERROR", "EXITED", "TERMINATED"):
            raise RuntimeError(f"Pod ended up {status}")
        for p in ((pod.get("runtime") or {}).get("ports")) or []:
            if str(p.get("private")) == "22" and p.get("ip") and p.get("public"):
                log(f"Pod RUNNING, SSH at {p['ip']}:{p['public']}")
                return p["ip"], int(p["public"])
        log(f"… pod status: {status}")
        time.sleep(15)
    raise TimeoutError("Pod did not become ready in time")


# ------------------------------------------------------------------------ SSH
def ssh_connect(ip, port):
    for attempt in range(1, 21):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # pods are throwaway
            client.connect(ip, port=port, username="root", key_filename=str(SSH_KEY),
                           look_for_keys=False, allow_agent=False, timeout=20)
            log("🔑 SSH connected")
            return client
        except Exception as e:
            log(f"… SSH not ready yet ({type(e).__name__}), retry {attempt}/20")
            time.sleep(15)
    raise RuntimeError("Could not connect over SSH")


def ssh_run(client, command, timeout=1800):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode(errors="ignore").strip(), stderr.read().decode(errors="ignore").strip()


def install_models(client):
    _, ckpt, _ = ssh_run(
        client,
        "find / -maxdepth 6 -type d -path '*ComfyUI/models/checkpoints' 2>/dev/null | head -1",
        timeout=120)
    if not ckpt:
        raise RuntimeError("Couldn't find ComfyUI's models folder on the pod")
    models_root = ckpt.rsplit("/", 1)[0]
    log(f"Models folder on pod: {models_root}")

    jobs = []
    for sub, name, url, _ in MODEL_FILES:
        path = f"{models_root}/{sub}/{name}"
        jobs.append(f"( mkdir -p '{models_root}/{sub}' && ( [ -s '{path}' ] || "
                    f"( curl -sSL --retry 3 -o '{path}.part' '{url}' && mv '{path}.part' '{path}' ) ) )")
    log("⬇ Downloading Z-Image Turbo (~21 GB) + upscaler onto the pod (usually 2-6 min)...")
    _, _, err = ssh_run(client, " & ".join(jobs) + " & wait", timeout=2400)

    for sub, name, _, min_size in MODEL_FILES:
        _, size, _ = ssh_run(client, f"stat -c %s '{models_root}/{sub}/{name}' 2>/dev/null || echo 0")
        if int(size or 0) < min_size:
            raise RuntimeError(f"{name} did not download properly. Details: {err[:300]}")
    log("✅ Models ready")


# ------------------------------------------------------------------- ComfyUI
def comfy_url(pod_id):
    return f"https://{pod_id}-8188.proxy.runpod.net"


def wait_for_comfy(base, timeout=COMFY_START_TIMEOUT_MIN * 60):
    deadline = time.time() + timeout
    while time_left() > 0 and time.time() < deadline:
        try:
            r = requests.get(f"{base}/system_stats", timeout=15)
            if r.ok:
                devices = r.json().get("devices", [])
                log(f"✅ ComfyUI is up ({devices[0].get('name', '?') if devices else '?'})")
                return
        except requests.RequestException:
            pass
        log("… waiting for ComfyUI")
        time.sleep(15)
    raise TimeoutError(f"ComfyUI did not start within {timeout // 60} min (faulty GPU/pod?)")


def build_workflow(prompt, seed, prefix):
    """Z-Image Turbo text-to-image + 2x RealESRGAN upscale, in ComfyUI API format."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": STEPS, "cfg": 1.0,
                         "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x2plus.pth"}},
        "11": {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["10", 0], "image": ["9", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


def run_job(base, wf, timeout=900):
    r = requests.post(f"{base}/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())}, timeout=60)
    if not r.ok:
        raise RuntimeError(f"ComfyUI rejected the job: {r.status_code} {r.text[:400]}")
    prompt_id = r.json()["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline and time_left() > 0:
        h = requests.get(f"{base}/history/{prompt_id}", timeout=30)
        if h.ok and prompt_id in h.json():
            entry = h.json()[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                detail = msgs[0][1].get("exception_message", "")[:300] if msgs else ""
                raise RuntimeError(f"ComfyUI error: {detail}")
            images = entry.get("outputs", {}).get("12", {}).get("images", [])
            if images:
                return images
        time.sleep(3)
    raise TimeoutError("Image took too long")


# ------------------------------------------------------- launch a healthy pod
def gpu_health(ssh):
    """Fail fast: check the host driver and that PyTorch inside ComfyUI's own environment can use the GPU."""
    code, out, err = ssh_run(ssh, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader", timeout=60)
    if code != 0 or not out:
        raise RuntimeError(f"nvidia-smi failed: {(err or out)[:200]}")
    _, cuda, _ = ssh_run(ssh, "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'", timeout=60)
    log(f"GPU visible: {out} | host {cuda or 'CUDA version unknown'}")

    find_py = ("PY=$(ls -d /workspace/*/ComfyUI/.venv*/bin/python 2>/dev/null | head -1); "
               "[ -z \"$PY\" ] && PY=$(find / -maxdepth 6 -path '*ComfyUI/.venv*/bin/python' 2>/dev/null | head -1); echo $PY")
    _, py, _ = ssh_run(ssh, find_py, timeout=120)
    if not py:
        log("… couldn't find ComfyUI's Python yet - relying on the ComfyUI start-up check instead")
        return
    test = (f"{py} -c \"import torch; torch.cuda.init(); "
            f"print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))\"")
    code, out, err = ssh_run(ssh, test, timeout=180)
    if code != 0:
        last = (err or out).strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"PyTorch can't use this GPU: {last[0][:200]}")
    log(f"✅ GPU works in ComfyUI's Python: {out}")


def launch_ready_pod():
    """Create pods until one has a working GPU and a running ComfyUI. Faulty pods are terminated."""
    excluded = set()
    for attempt in range(1, MAX_POD_ATTEMPTS + 1):
        pod_id, gpu, cloud = create_pod(excluded)
        if not pod_id:
            return None
        ssh = None
        try:
            ip, port = wait_for_ssh_port(pod_id)
            ssh = ssh_connect(ip, port)
            gpu_health(ssh)
            base = comfy_url(pod_id)
            wait_for_comfy(base)
            return pod_id, gpu, cloud, ssh, base
        except BaseException as e:
            if ssh:
                ssh.close()
            terminate(pod_id)
            if isinstance(e, KeyboardInterrupt):
                raise
            excluded.add(gpu)
            log(f"✖ Attempt {attempt}: {gpu} pod was faulty ({e}). Trying a different GPU type...")
    log("✖ Gave up after several faulty pods. Nothing more will be charged.")
    return None


# ---------------------------------------------------------------------- main
def main():
    prompts = preflight()
    total = len(prompts) * IMAGES_PER_PROMPT
    out_dir = Path(OUTPUT_ROOT) / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"{len(prompts)} prompts × {IMAGES_PER_PROMPT} = {total} images → {out_dir}")

    try:
        ready = launch_ready_pod()
    except KeyboardInterrupt:
        log("Ctrl+C pressed - nothing left running.")
        return
    if not ready:
        log("No working GPU available right now. Try again later.")
        return
    pod_id, gpu, cloud, ssh, base = ready

    saved, failures_in_a_row = 0, 0
    try:
        install_models(ssh)

        batch_tag = datetime.now().strftime("%Y%m%d_%H%M")
        with open(out_dir / "log.csv", "w", newline="", encoding="utf-8") as f, \
             open(out_dir / f"adobe_upload_{batch_tag}.csv", "w", newline="", encoding="utf-8") as fa:
            writer = csv.writer(f)
            writer.writerow(["file", "theme", "title", "prompt", "seed", "width", "height"])
            adobe = csv.writer(fa, quoting=csv.QUOTE_ALL)
            adobe.writerow(["Filename", "Title", "Keywords", "Category", "Releases"])
            n = 0
            for row in prompts:
                for v in range(1, IMAGES_PER_PROMPT + 1):
                    n += 1
                    if time_left() < 180:
                        log("⏱ Time limit nearly reached - stopping the batch early.")
                        raise StopIteration
                    seed = random.randint(1, 2**48)
                    file_name = f"{FILE_PREFIX}_{batch_tag}_{n:03d}.jpg"   # e.g. LL_20261117_1600_001.jpg
                    try:
                        t0 = time.time()
                        images = run_job(base, build_workflow(row["prompt"], seed, f"{FILE_PREFIX}_{n:03d}"))
                        img = images[0]
                        data = requests.get(f"{base}/view", params={
                            "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")}, timeout=300)
                        data.raise_for_status()
                        picture = Image.open(io.BytesIO(data.content)).convert("RGB")
                        picture.save(out_dir / file_name, "JPEG", quality=JPEG_QUALITY, optimize=True)
                        w, h = picture.size
                        writer.writerow([file_name, row.get("theme", ""), row["title"], row["prompt"], seed, w, h])
                        adobe.writerow([file_name, row["title"].strip(), row["keywords"].strip(),
                                        row["category"].strip(), ""])
                        f.flush(); fa.flush()
                        saved += 1
                        failures_in_a_row = 0
                        log(f"🖼 {n}/{total} {file_name} {w}x{h} ({w*h/1e6:.1f} MP, "
                            f"{time.time() - t0:.0f}s) - {row['title'][:45]}")
                    except Exception as e:
                        failures_in_a_row += 1
                        log(f"✖ {n}/{total} failed: {e}")
                        if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES:
                            log("✖ Several failures in a row - stopping so you don't pay for a broken run.")
                            raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        log("Ctrl+C pressed - cleaning up.")
    except Exception as e:
        log(f"✖ Batch stopped: {e}")
    finally:
        if ssh:
            ssh.close()
        terminate(pod_id)
        log(f"Done: {saved}/{total} images saved to {out_dir} | {gpu} ({cloud}) | "
            f"~{(time.time() - START) / 60:.1f} min total")


if __name__ == "__main__":
    main()
