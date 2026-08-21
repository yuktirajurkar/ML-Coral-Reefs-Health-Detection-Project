from pathlib import Path
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm
import pandas as pd

from config import TRAIN_DIR, VALIDATION_DIR, TEST_DIR

# -------------------------
# Blur Threshold
# -------------------------
BLUR_THRESHOLD = 100

# -------------------------
# Collect all image paths
# -------------------------
image_paths = []

for folder in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    image_paths.extend(folder.rglob("*.jpg"))
    image_paths.extend(folder.rglob("*.jpeg"))
    image_paths.extend(folder.rglob("*.png"))

print(f"\nTotal Images Found : {len(image_paths)}")

# -------------------------
# Validation Results
# -------------------------
results = []

seen_hashes = {}

for img_path in tqdm(image_paths):

    record = {}

    record["image"] = img_path.name
    record["path"] = str(img_path)

    # Class Label
    if "healthy" in str(img_path).lower():
        record["label"] = "Healthy"
    else:
        record["label"] = "Bleached"

    # -------------------------
    # Read Image
    # -------------------------
    img = cv2.imread(str(img_path))

    if img is None:
        record["status"] = "Corrupted"
        results.append(record)
        continue

    record["status"] = "OK"

    h, w = img.shape[:2]

    record["height"] = h
    record["width"] = w

    # -------------------------
    # Blur Detection
    # -------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    record["blur_score"] = round(blur_score, 2)

    if blur_score < BLUR_THRESHOLD:
        record["blurry"] = "Yes"
    else:
        record["blurry"] = "No"

    # -------------------------
    # Duplicate Detection
    # -------------------------

    hash_value = imagehash.average_hash(Image.open(img_path))

    if hash_value in seen_hashes:
        record["duplicate"] = "Yes"
    else:
        record["duplicate"] = "No"
        seen_hashes[hash_value] = img_path

    results.append(record)

# -------------------------
# Save Report
# -------------------------

df = pd.DataFrame(results)

print(df.head())

df.to_csv("../outputs/reports/validation_report.csv", index=False)

print("\nValidation Report Saved Successfully.")
print("\n========== VALIDATION SUMMARY ==========\n")

print("Total Images :", len(df))
print("Corrupted Images :", (df["status"] == "Corrupted").sum())
print("Blurry Images :", (df["blurry"] == "Yes").sum())
print("Duplicate Images :", (df["duplicate"] == "Yes").sum())

print("\nClass Distribution:\n")
print(df["label"].value_counts())

print("\nImage Dimensions:\n")
print(df[["width", "height"]].describe())