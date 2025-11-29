import os
import sys
from pathlib import Path
# Add submodules/mle-dojo to the Python path
mle_dojo_path = Path(__file__).parent / "submodules" / "mle-dojo"
sys.path.insert(0, str(mle_dojo_path))
# from mledojo.gym.competition import CompetitionRegistry, CompInfo
# from mledojo.competitions import get_metric
# from mledojo.gym.env import KaggleEnvironment

import os
from pathlib import Path
from mledojo.gym.competition import CompetitionRegistry, CompInfo, Competition
from mledojo.competitions import get_metric
from mledojo.gym.interface import (
    Interface,
    InfoInterface, 
    CodeValidationInterface, 
    CodeExecutionInterface
)
from mledojo.gym.sandbox import Sandbox
from mledojo.gym.env import KaggleEnvironment
from mledojo.gym.feedback import FeedbackManager, Feedback
from mledojo.utils import get_metric






competition_name = "home-data-for-ml-course"
data_dir = "/home/rklotins/src/mle-RL/submodules/mle-dojo/data/prepared/home-data-for-ml-course/data"
output_dir = "output/home-data-for-ml-course"

# register the competition
registry = CompetitionRegistry()
registry.register(
    name=competition_name,
    data_dir=data_dir,  # "random-acts-of-pizza/data"
    comp_info=CompInfo(
        category="General",
        level="beginner",
        output_type="submission.csv",
        higher_is_better=True
    ),
    metric_class=get_metric(competition_name)
)


# initialize the env
env = KaggleEnvironment.make(
    competition_name=competition_name,      
    output_dir=output_dir,
    competition_registry=registry,                  
    score_mode="position",              
    gpu_device=0,                     
    gpu_memory_limit=32,                   
    execution_timeout=3600             
)


# request_info
env.step("request_info", **{"info_type": "overview"})

# validate_code
env.step("validate_code", **{"code": "import pandas as pd\nprint('Welcome to MLE-Dojo!')"})

# Execute_code
absolute_data_dir = Path(os.path.abspath(data_dir))
absolute_output_dir = Path(os.path.abspath(output_dir))
code_to_execute = f'''
import pandas as pd
submission = pd.read_csv('{absolute_data_dir / "public" / "sample_submission.csv"}')
submission.to_csv('{absolute_output_dir / "submission.csv"}', index=False)
print("Submission created successfully.")
'''
env.step("execute_code", **{"code": code_to_execute})


feedback_manager = FeedbackManager()
exec_context = {
    "score_mode": "position",  # or "raw"
    "best_raw_score": 0.75, # Example best score
    "best_position_score": 0.9 # Example best score
}
