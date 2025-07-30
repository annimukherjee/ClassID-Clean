# Step 1: Choose a suitable base image. Python 3.10-slim is a good balance.
FROM python:3.10-slim

# Step 2: Set the working directory inside the container.
# This will be the root of your project.
WORKDIR /app

# Step 3: Install system-level dependencies.
# These are commonly required for libraries like OpenCV and other video processing tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy ONLY the requirements file to leverage Docker's layer caching.
# Note the name change to a standard 'requirements.txt'.
COPY requirements_claddid-reid_conda.txt ./requirements.txt

# Step 5: Install the Python dependencies using pip.
# The --no-cache-dir flag keeps the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application's code into the container.
# This will be handled by the .dockerignore file to exclude unnecessary files.
COPY . .

# Step 7: Set the default command to open a bash shell.
# This is perfect for a multi-step workflow, allowing the user to
# run commands interactively.
CMD [ "/bin/bash" ]