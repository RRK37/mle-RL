FROM python:3.11-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY . /app

# --- FIX START ---

# 1. Upgrade pip/wheel so it can find pre-built binaries (Critical!)
RUN pip install --upgrade pip setuptools wheel

# 2. (Optional but recommended) Install heavy dependencies explicitly first.
# This prevents the 'mle-dojo' install from trying to compile them.
RUN pip install numpy pandas

# 3. Now install your package with verbose output (-v)
# The -v flag will print what is happening so you aren't staring at a frozen screen.
RUN pip install -v -e ./submodules/mle-dojo


CMD ["python", "first_agent.py"]