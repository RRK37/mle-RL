# MLE-RL Agent Runner

This is a wrapper for running machine learning agents on Kaggle competitions using the MLE-Dojo framework.

## Setup

1. Make sure you have the `mle-dojo` submodule initialized:
   ```bash
   git submodule update --init --recursive
   ```

2. Install required dependencies (from the mle-dojo submodule):
   ```bash
   cd submodules/mle-dojo
   pip install -e .
   cd ../..
   ```

3. Set up your OpenAI API key (if using AIDE, OpenAI, or MLE agents with OpenAI):
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Using the Config File

The easiest way to run an agent is using the configuration file:

```bash
python main.py --config config.yaml
```

### Config File Structure

Edit `config.yaml` to customize your run:

- **agent_type**: Choose from `aide`, `mle`, `openai`, or `dummy`
- **competition.name**: Name of the Kaggle competition (e.g., `titanic`, `house-prices`)
- **competition.data_dir**: Path to competition data directory
- **output_dir**: Where to save results and logs
- **env.execution_timeout**: Maximum runtime in seconds
- **agent.steps**: Number of improvement iterations (for AIDE agent)
- **agent.code.model_name**: LLM model to use (e.g., `gpt-4o-mini`, `gpt-4`)

### Command Line Arguments

You can also run with command line arguments:

```bash
python main.py \
  --agent-type aide \
  --competition-name titanic \
  --data-dir ./submodules/mle-dojo/data \
  --output-dir ./output \
  --execution-timeout 43200 \
  --max-steps 15
```

### Examples

#### Running AIDE Agent on Titanic

```bash
# Edit config.yaml to set:
# agent_type: aide
# competition.name: titanic

python main.py --config config.yaml
```

#### Running with Different Competitions

```bash
# For house prices competition
python main.py \
  --config config.yaml \
  --competition-name house-prices-advanced-regression-techniques
```

#### Using Environment Variable for Competition Name

```bash
export COMPETITION_NAME=titanic
python main.py --config config.yaml
```

## Output

Results are saved to the output directory specified in the config:

- **output/{competition_name}/**: Main output directory
  - **logs/**: AIDE agent logs and visualizations
  - **workspaces/**: Agent workspaces with generated code
  - **agent_history/**: Trajectory and cost history for MLE/OpenAI agents

## Agent Types

### AIDE Agent
- Tree-search based agent with iterative improvement
- Best for: Complex competitions requiring exploration
- Key settings: `steps`, `search.num_drafts`, `search.max_debug_depth`

### MLE Agent
- Sequential reasoning agent
- Best for: Straightforward competitions
- Key settings: `max_steps`, `model_name`

### OpenAI Agent
- Direct OpenAI API integration
- Best for: Testing with latest OpenAI models
- Key settings: `max_steps`, `model_name`

### Dummy Agent
- Simple baseline agent
- Best for: Testing environment setup

## Troubleshooting

### Import Errors
Make sure you've installed the mle-dojo package and are running from the project root:
```bash
cd /path/to/mle-RL
python main.py --config config.yaml
```

### Data Not Found
Ensure competition data is in the correct location:
```
submodules/mle-dojo/data/prepared/{competition_name}/data/
```

If the data is not prepared yet, you may need to prepare it first using the mle-dojo preparation scripts.

### API Key Issues
Verify your OpenAI API key is set:
```bash
echo $OPENAI_API_KEY
```

## Configuration Files Location

- **Root config**: `./config.yaml` (this file, customizable)
- **AIDE agent config**: `./submodules/mle-dojo/mledojo/agent/aide/utils/config.yaml`
- **MLE agent config**: `./submodules/mle-dojo/mledojo/agent/mleagent/config.yaml`
- **OpenAI agent config**: `./submodules/mle-dojo/mledojo/agent/openaiagent/config.yaml`

The root `config.yaml` overrides agent-specific settings when provided.
