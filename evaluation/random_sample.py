import numpy as np
import argparse


# Read a single int from args
parser = argparse.ArgumentParser()
parser.add_argument("N", type=int)

args = parser.parse_args()

# Generate a random float array of size N
arr = np.random.uniform(size=args.N).astype(np.float32)

# Write the array to a file
with open("random.bin", "wb") as f:
    f.write(arr.tobytes())
