# Use an official NVIDIA CUDA runtime as a parent image.
# This image comes with CUDA and cuDNN pre-installed, which are necessary for GPU support.
# We use a -devel image to get the full CUDA toolkit, which can be useful for compiling
# dependencies that require it.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container to /app
WORKDIR /app

# Set environment variables to ensure that package installations are non-interactive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install necessary system dependencies.
# - python3.10 and python3-pip are needed to run the Python application.
# - git is required because some Python packages are installed from git repositories.
# - libsndfile1 is a library for reading/writing audio files, a common dependency for audio packages.
# - ffmpeg is a widely used utility for audio and video processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project from your local machine to the /app directory in the container
COPY . .

# Install the Python dependencies for the project.
# We first upgrade pip to the latest version.
# Then, we install the project itself. `pip install .` will read the pyproject.toml
# and install all the specified dependencies.
# We also install nltk, which is used for sentence tokenization in the long-form generation notebook.
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir .
RUN pip3 install --no-cache-dir nltk

# Download the 'punkt' tokenizer data from nltk, which is required for sentence splitting.
RUN python3 -m nltk.downloader punkt

# Set environment variables for GPU usage and for the Hugging Face cache directory.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/huggingface_cache

# The following environment variables can be uncommented to optimize for lower VRAM GPUs.
# ENV SUNO_USE_SMALL_MODELS=True
# ENV SUNO_OFFLOAD_CPU=True

# Pre-load the Bark models during the Docker image build.
# This will cache the models within the image, leading to faster startup times when you run the container.
RUN python3 -c "from bark.generation import preload_models; preload_models()"

# Define the default command to run when the container starts.
# This command uses the Bark command-line interface to generate a sample audio file.
# The output file will be saved to /app/bark_generation.wav inside the container.
CMD ["python3", "-m", "bark", "--text", "Hello, my name is Suno and I like pizza. [laughs]", "--output_filename", "/app/bark_generation.wav"]
