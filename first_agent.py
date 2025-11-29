import sys
from pathlib import Path
# Add submodules/mle-dojo to the Python path
mle_dojo_path = Path(__file__).parent / "submodules" / "mle-dojo"
sys.path.insert(0, str(mle_dojo_path))
import os
from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
from mledojo.gym.env import KaggleEnvironment

# 1. SETUP CONFIG
# ----------------
competition_name = "titanic"
# Use absolute paths to be safe on Windows
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "data", competition_name)
output_dir = os.path.join(current_dir, "output", competition_name)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 2. REGISTER THE COMPETITION
# ---------------------------
# MLE-Dojo needs to know this competition exists and where the data is.
registry = CompetitionRegistry()
registry.register(
    name=competition_name,
    data_dir=data_dir,
    comp_info=CompInfo(
        category="Tabular",
        level="beginner",
        output_type="submission.csv",
        higher_is_better=True
    ),
    metric_class=get_metric(competition_name) # Uses standard accuracy metric
)

print(f"✅ Registered {competition_name}")

# 3. START THE DOJO (This launches the Docker Container)
# ----------------------------------------------------
print("📦 Spinning up the environment (this might take a minute)...")
env = KaggleEnvironment.make(
    competition_name=competition_name,
    output_dir=output_dir,
    competition_registry=registry,
    score_mode="position",
    gpu_device=0,           # Set to -1 if you don't have a GPU
    execution_timeout=600   # 10 minute limit
)
print("✅ Environment Ready!")

# 4. RUN AN EXPERIMENT
# --------------------
# A. Get Info about the task
obs, info = env.reset()
print("\n--- 1. TASK INFO ---")
# Request task info explicitly since it's not in reset() observation
obs, reward = env.step("request_info", info_type="all")
# The instruction is likely in the feedback or result of this step
print(f"Info Request Result: {obs}")

# B. Write a 'Hello World' solution
# This is what your Agent/LLM would generate.
# We will write a simple script that just predicts everyone dies (classic baseline).
agent_code = f"""
import pandas as pd
import os

# 1. Load Data
# The Docker container mounts data at a specific path provided by the env
train = pd.read_csv('{data_dir.replace(os.sep, '/')}/train.csv')
test = pd.read_csv('{data_dir.replace(os.sep, '/')}/test.csv')

# 2. Simple Logic: Everyone dies (0)
submission = pd.DataFrame({{
    'PassengerId': test['PassengerId'],
    'Survived': 0
}})

# 3. Save Submission
# The env expects us to save to 'submission.csv' in the output dir
output_path = '{output_dir.replace(os.sep, '/')}/submission.csv'
submission.to_csv(output_path, index=False)
print(f"Saved submission to {{output_path}}")
"""

print("\n--- 2. EXECUTING AGENT CODE ---")
# The 'execute_code' step runs the python string INSIDE the Docker container
# NOTE: env.step() returns (obs, reward) in this version of mle-dojo, despite type hints saying otherwise
obs, reward = env.step("execute_code", code=agent_code)

print("\n--- 3. RESULTS ---")
print(f"Observation: {obs}")
print(f"🏆 Reward (HumanRank): {reward}")