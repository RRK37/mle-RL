from mledojo.gym.competition import CompetitionRegistry, CompInfo
from mledojo.competitions import get_metric
from mledojo.gym.env import KaggleEnvironment

competition_name = "random-acts-of-pizza"
data_dir = ...
output_dir = ...

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
