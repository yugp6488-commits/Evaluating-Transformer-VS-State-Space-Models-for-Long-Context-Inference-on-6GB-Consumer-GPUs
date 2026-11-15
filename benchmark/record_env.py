import os
import sys
from benchmark.system_monitor import log_system_metadata, print_system_info

print_system_info()
os.makedirs("results/final_experiment", exist_ok=True)
p1 = log_system_metadata("results/final_experiment/environment_metadata.csv")
p2 = log_system_metadata("environment_metadata.csv")
print(f"Logged metadata to:\n  - {p1}\n  - {p2}")
