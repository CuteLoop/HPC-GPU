"""
Generate a uniform histogram dataset: all elements map to the same bin (bin 42).
This stresses atomic contention — every thread hammers the same bin.
Output format matches dataset_generator.cpp: first line = count, then one value per line.
"""
import os

hw_path = os.path.dirname(os.path.abspath(__file__))
uniform_dir = os.path.join(hw_path, "Histogram", "Dataset", "uniform")
os.makedirs(uniform_dir, exist_ok=True)

NUM_ELEMENTS = 500000
NUM_BINS = 4096
BIN_CAP = 127
VALUE = 42  # all elements map to bin 42

# Write input.raw
input_path = os.path.join(uniform_dir, "input.raw")
with open(input_path, "w") as f:
    f.write(str(NUM_ELEMENTS))
    for _ in range(NUM_ELEMENTS):
        f.write(f"\n{VALUE}")

# Compute expected output (with clipping at 127)
bins = [0] * NUM_BINS
count = NUM_ELEMENTS
bins[VALUE] = min(count, BIN_CAP)

# Write output.raw
output_path = os.path.join(uniform_dir, "output.raw")
with open(output_path, "w") as f:
    f.write(str(NUM_BINS))
    for b in bins:
        f.write(f"\n{b}")

print(f"Generated uniform dataset: {NUM_ELEMENTS} elements, all = {VALUE}")
print(f"  input.raw  -> {input_path}")
print(f"  output.raw -> {output_path}")
print(f"  bin[{VALUE}] = {bins[VALUE]} (clipped from {NUM_ELEMENTS}), all other bins = 0")
