
import pandas as pd
import os

# 1. Load Data
# The Docker container mounts data at a specific path provided by the env
train = pd.read_csv('/app/data/titanic/train.csv')
test = pd.read_csv('/app/data/titanic/test.csv')

# 2. Simple Logic: Everyone dies (0)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': 0
})

# 3. Save Submission
# The env expects us to save to 'submission.csv' in the output dir
output_path = '/app/output/titanic/submission.csv'
submission.to_csv(output_path, index=False)
print(f"Saved submission to {output_path}")
