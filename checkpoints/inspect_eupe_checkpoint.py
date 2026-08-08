import torch
from pathlib import Path

checkpoint_path = Path(
    r"C:\Users\lenovo\.cache\huggingface\hub\models--facebook--EUPE-ViT-S"
)

# Find the downloaded .pt file automatically
pt_files = list(checkpoint_path.rglob("EUPE-ViT-S.pt"))

if not pt_files:
    raise FileNotFoundError("EUPE-ViT-S.pt not found")

checkpoint_path = pt_files[0]

print("Checkpoint:")
print(checkpoint_path)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
)

print("\nCheckpoint type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nTop-level keys:")
    for key in checkpoint.keys():
        print(" ", key)

    print("\nNumber of top-level entries:")
    print(len(checkpoint))
else:
    print("\nCheckpoint is not a dictionary.")