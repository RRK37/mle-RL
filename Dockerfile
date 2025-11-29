FROM python:3.11-slim

# Install system build dependencies
# 'slim' images lack compilers. We add build-essential for compiling C extensions
# and git for checking out code.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- OPTIMIZED LAYER CACHING ---

# Upgrade pip first (standard practice)
RUN pip install --upgrade pip setuptools wheel

# Install heavy dependencies *before* copying your code
# This way, if you change your script, Docker uses the cached layer
# and doesn't re-download pandas/numpy
RUN pip install numpy pandas

# Handle the Submodule
# Copy ONLY the submodule directory first to install it
# IMPORTANT: Run 'git submodule update --init submodules/mle-dojo' before building!
COPY ./submodules/mle-dojo /app/submodules/mle-dojo

# Install the mle-dojo package in editable mode
RUN pip install -v -e ./submodules/mle-dojo

# Copy application scripts
# This layer changes most often, so it's last for optimal caching
COPY *.py /app/

# Create directories for data and output
# These can be mounted as volumes at runtime
RUN mkdir -p /app/data /app/output

# IMPORTANT: Data should be mounted as a volume when running the container
# Example: docker run -v ./data:/app/data -v ./output:/app/output mle-rl

# Set environment variables
ENV PYTHONUNBUFFERED=1

CMD ["python", "first_agent.py"]