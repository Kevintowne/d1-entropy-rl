import os
from datasets import load_dataset
out = os.environ.get("OUT","./cache/s1K")
ds = load_dataset("simplescaling/s1K")
ds.save_to_disk(out)
print("saved ->", out)
