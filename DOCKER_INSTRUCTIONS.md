# Docker Instructions for MLE-RL

This guide explains how to build and run the MLE-RL project using Docker.

## Prerequisites

Before building the Docker image, you **must** initialize the mle-dojo submodule:

```bash
git submodule update --init submodules/mle-dojo
```

This ensures that the `submodules/mle-dojo` directory contains the required code without initializing the `mle-bench` submodule.

## Prepare Data

Make sure you have your competition data in the correct location. For the Titanic example:

```
mle-RL/
├── data/
│   └── titanic/
│       ├── train.csv
│       └── test.csv
└── ...
```

If you don't have the data yet, download it from Kaggle or your competition source.

### 2. Restructure Data for MLE-Dojo

MLE-Dojo requires a specific directory structure (public/private split). We have provided a script to set this up for you.

Run this command inside the Docker container (or locally if you have pandas installed):

```bash
# Run inside Docker (Recommended)
docker run -v ${PWD}:/app mle-rl python prepare_data.py
```

This will create `data/titanic/public` and `data/titanic/private` folders required by the framework.

## Build the Docker Image

Build the image with:

```bash
docker build -t mle-rl .
```

**Build arguments:**
- `-t mle-rl` tags the image with the name "mle-rl"
- `.` specifies the build context (current directory)

## Run the Container

### Basic Run (with volume mounts + Kaggle credentials)

**RECOMMENDED:** Mount your local `data`, `output`, and Kaggle credentials to avoid 401 errors:

```bash
docker run -v ./data:/app/data -v ./output:/app/output -v ~/.kaggle:/root/.kaggle mle-rl
```

### Windows PowerShell

On Windows PowerShell, use full paths with Kaggle credentials:

```powershell
docker run -v ${PWD}/data:/app/data -v ${PWD}/output:/app/output -v ${HOME}/.kaggle:/root/.kaggle mle-rl
```

> [!IMPORTANT]
> The `-v ${HOME}/.kaggle:/root/.kaggle` mount is essential to avoid **401 Unauthorized errors** from the mle-dojo framework when it tries to authenticate with Kaggle APIs.

### Interactive Mode

To run the container interactively (useful for debugging):

```bash
docker run -it -v ./data:/app/data -v ./output:/app/output mle-rl /bin/bash
```

Then inside the container, you can run:

```bash
python first_agent.py
```

### Run Different Scripts

To run a different Python script:

```bash
docker run -v ./data:/app/data -v ./output:/app/output mle-rl python quickStart.py
```

## GPU Support (Optional)

If you have NVIDIA GPU support and want to use it:

```bash
docker run --gpus all -v ./data:/app/data -v ./output:/app/output mle-rl
```

Note: This requires:
- NVIDIA Docker runtime installed
- Appropriate GPU drivers
- Update `gpu_device=0` in your Python script (already set in `first_agent.py`)

## Environment Variables

You can pass environment variables to customize behavior:

```bash
docker run -e COMPETITION_NAME=titanic -v ./data:/app/data -v ./output:/app/output mle-rl
```

## Troubleshooting

### 401 Unauthorized / Client Error

**Problem:** Container shows `401 Client Error: Unauthorized` when running.

**Solution:** The mle-dojo framework needs Kaggle API credentials. You have two options:

**Option A: Mount credentials (Recommended)**
```powershell
docker run -v ${PWD}/data:/app/data -v ${PWD}/output:/app/output -v ${HOME}/.kaggle:/root/.kaggle mle-rl
```

**Option B: Pass credentials as environment variables**
```powershell
docker run -e KAGGLE_USERNAME=your_username -e KAGGLE_KEY=your_api_key -v ${PWD}/data:/app/data -v ${PWD}/output:/app/output mle-rl
```

To get your Kaggle credentials:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" (downloads `kaggle.json`)
4. Place it in `~/.kaggle/` (or `%USERPROFILE%/.kaggle/` on Windows)

### "No such file or directory" errors for submodules

**Problem:** The `mle-dojo` submodule is empty.

**Solution:** Run `git submodule update --init submodules/mle-dojo` before building.

### Data files not found

**Problem:** The container can't find `train.csv` or `test.csv`.

**Solution:** 
- Ensure data is in the correct path: `./data/titanic/`
- Mount the volume correctly: `-v ./data:/app/data`
- On Windows, use absolute paths: `-v C:\Dev\iterate_hack\mle-RL\data:/app/data`

### Permission errors on output directory

**Problem:** Container can't write to output directory.

**Solution:** Create the output directory first and ensure it has write permissions:

```bash
mkdir -p output/titanic
chmod 777 output  # Unix/Linux/Mac
```

### Docker build is slow

**Problem:** Each build takes a long time.

**Solution:** 
- The Dockerfile is optimized for layer caching
- Don't change `requirements.txt` or submodules frequently
- Use `.dockerignore` to exclude unnecessary files (already included)

## Clean Up

Remove the image:

```bash
docker rmi mle-rl
```

Remove all stopped containers:

```bash
docker container prune
```

## Next Steps

1. Modify `first_agent.py` to implement your RL agent
2. Add more competitions to the `data/` directory
3. Run experiments and collect results from `output/`
