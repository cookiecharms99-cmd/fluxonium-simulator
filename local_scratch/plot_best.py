import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# maybe there's a way to use the sha's but im dumb so manual for now!
sys.path.insert(0, str(PROJECT_ROOT))
data_folder = "reference_6h_legacy_4776599"


data_folder_path = PROJECT_ROOT / "data" / "output" / "cmaes" / data_folder

metadata_path = data_folder_path / "metadata.json"
with open(metadata_path, "r") as f:
    metadata = json.load(f)
config_file = metadata["run_parameters"]["physics_config_path"]
physics_config_fp = PROJECT_ROOT / config_file
duration = 150  # perhaps we need a record of this eventually :]
with open(physics_config_fp, "r") as f:
    physics_config = json.load(f)
time_array = np.linspace(0, duration, 500)

print("Controls!")
with np.load(data_folder_path / "controls.npz") as data:
    time_array = data["times"]
    flux_array = data["external_phase"]
    charge_drive = data["charge_drive"]

plt.plot(time_array, flux_array)
plt.show()
