import os
import config

print("Current working directory:", os.getcwd())
print("Dataset path:", config.DATASET_PATH)
print("Exists:", os.path.exists(config.DATASET_PATH))