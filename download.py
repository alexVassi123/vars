import os
import zipfile
import time
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

mySNdl = SNdl(LocalDirectory="data/SoccerNet")

for split in ["train", "valid", "test", "challenge"]:
    for attempt in range(5):
        try:
            print(f"Downloading {split} (attempt {attempt+1})...")
            mySNdl.downloadDataTask(task="mvfouls", split=[split], password="s0cc3rn3t")
            break
        except Exception as e:
            print(f"Failed: {e}. Retrying in 5s...")
            time.sleep(5)


splits = {
    "train": "Train",
    "valid": "Valid",
    "test": "Test",
    "challenge": "Chall",
}

base = "data/SoccerNet/mvfouls"

for zip_name, folder_name in splits.items():
    zip_path = os.path.join(base, f"{zip_name}.zip")
    out_path = os.path.join(base, folder_name)
    os.makedirs(out_path, exist_ok=True)
    print(f"Extracting {zip_path} -> {out_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_path)

print("Done. Dataset ready.")
