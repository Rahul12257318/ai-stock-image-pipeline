"""
Lumen & Lane - prepare an Adobe Stock upload from your hand-picked images.

How to use:
  1. Review your batch folder(s) in output\\ and COPY the images you want to upload into:
         output\\Final selected\\
     (you can pick from several different batches - that's fine)
  2. Run:   python prepare_upload.py
  3. Upload the JPEGs in "Final selected" to Adobe, then upload the CSV this script
     writes into that same folder:  adobe_upload_FINAL_<date-time>.csv

The script finds each image's title/keywords/category in the batch it came from,
and prints whether to mark it as Photo or Illustration in the Adobe portal.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_ROOT = Path("output")
FINAL_DIR = OUTPUT_ROOT / "Final selected"

# Adobe rule: submit as Photo only if it looks like a real camera photo of things that exist;
# everything else (3D renders, abstract art, holograms) as Illustration.
ILLUSTRATION_WORDS = ("3d render", "concept art", "abstract", "hologram", "illustration", "render style")
# Prompts showing a made-up interior, building or object -> tick "People and Property are fictional"
FICTIONAL_PROPERTY_WORDS = ("office", "interior", "room", "desk", "data center", "server", "podium")


def load_metadata():
    """Collect Adobe metadata + prompts from every batch folder (not from Final selected)."""
    adobe_rows, prompts = {}, {}
    for batch in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p != FINAL_DIR):
        for csv_file in batch.glob("adobe_upload_*.csv"):
            with open(csv_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    adobe_rows[row["Filename"]] = row
        log_file = batch / "log.csv"
        if log_file.exists():
            with open(log_file, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    prompts[row["file"]] = row.get("prompt", "")
    return adobe_rows, prompts


def main():
    if not OUTPUT_ROOT.is_dir():
        sys.exit("No output\\ folder here - run this from your pipeline folder.")
    if not FINAL_DIR.is_dir():
        FINAL_DIR.mkdir()
        sys.exit(f"Created '{FINAL_DIR}'. Copy the images you want to upload into it, then run this again.")

    images = sorted(p for p in FINAL_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if not images:
        sys.exit(f"No JPEGs in '{FINAL_DIR}' yet. Copy your chosen images in, then run this again.")

    adobe_rows, prompts = load_metadata()
    kept, missing = [], []
    for img in images:
        row = adobe_rows.get(img.name)
        if row:
            kept.append(row)
        else:
            missing.append(img.name)

    out = FINAL_DIR / f"adobe_upload_FINAL_{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "Title", "Keywords", "Category", "Releases"],
                                quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    print(f"\n✅ {len(kept)} images ready → {out}")
    if missing:
        print(f"⚠ {len(missing)} image(s) have no metadata (renamed, or not from a batch) - they are NOT in the CSV:")
        for name in missing:
            print(f"   - {name}")

    print("\nWhat to set for each image in the Adobe Contributor Portal:")
    print("  (ALL images: tick 'Created using generative AI tools')\n")
    for row in kept:
        p = prompts.get(row["Filename"], "").lower()
        looks_like_photo = "photograph" in p and not any(w in p for w in ILLUSTRATION_WORDS)
        ftype = "Photo" if looks_like_photo else "Illustration"
        extra = " + tick 'People and Property are fictional'" if any(w in p for w in FICTIONAL_PROPERTY_WORDS) else ""
        print(f"  {row['Filename']}  →  {ftype}{extra}   ({row['Title'][:45]})")
    print()


if __name__ == "__main__":
    main()
