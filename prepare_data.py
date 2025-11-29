import pandas as pd
import os
import shutil
from pathlib import Path

# Paths
base_dir = Path("data/titanic")
public_dir = base_dir / "public"
private_dir = base_dir / "private"

# Create directories
public_dir.mkdir(parents=True, exist_ok=True)
private_dir.mkdir(parents=True, exist_ok=True)

# 1. Load original train data
print("Loading original train.csv...")
df = pd.read_csv(base_dir / "train.csv")

# 2. Split into Train (80%) and Validation/Test (20%)
# We use the validation set as the "private test set" for the local dojo environment
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

print(f"Split data: Train={len(train_df)}, Test/Val={len(val_df)}")

# 3. Save Public Train Data
train_df.to_csv(public_dir / "train.csv", index=False)
print(f"Saved {public_dir / 'train.csv'}")

# 4. Save Public Test Data (Features only, no 'Survived')
test_df_public = val_df.drop(columns=["Survived"])
test_df_public.to_csv(public_dir / "test.csv", index=False)
print(f"Saved {public_dir / 'test.csv'}")

# 5. Save Private Test Answer (Ground Truth)
# mle-dojo usually expects the answer file to have the target column
val_df.to_csv(private_dir / "test_answer.csv", index=False)
print(f"Saved {private_dir / 'test_answer.csv'}")

# 6. Handle Sample Submission
# We'll base it on the test set we just created
sample_submission = pd.DataFrame({
    "PassengerId": val_df["PassengerId"],
    "Survived": 0
})
sample_submission.to_csv(public_dir / "sample_submission.csv", index=False)
print(f"Saved {public_dir / 'sample_submission.csv'}")

# 7. Create Description
desc_path = public_dir / "description.txt"
with open(desc_path, "w") as f:
    f.write("Titanic: Machine Learning from Disaster\n\nPredict survival on the Titanic.")
print(f"Saved {desc_path}")

print("\n✅ Data restructuring complete!")
