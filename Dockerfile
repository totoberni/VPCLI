# Stage 1: The Builder
# Use a specific version of Python for reproducibility
FROM python:3.12-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

# Copy the project definition file first for caching
COPY pyproject.toml .

# Copy the source code BEFORE installing the project
COPY src ./src

# Install the project and all its dependencies into the venv
RUN pip install --no-cache-dir .

# Stage 2: The Final Image
# Use the same base image
FROM python:3.12-slim AS final

# --- FIX: Install the 'unzip' utility ---
# We add this so we can extract our .zip archive.
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment (with dependencies and our app) from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the working directory
WORKDIR /app

# Activate the virtual environment for the ENTRYPOINT command
ENV PATH="/opt/venv/bin:$PATH"

# --- FIX: Copy and decompress assets from a single ZIP archive ---
COPY assets.zip .
RUN unzip assets.zip; rm assets.zip

# Set the environment variable to the now-extracted assets directory
ENV VPCLI_ASSETS_DIR=/app/assets

# Set the entrypoint for the CLI tool. This now exists in the PATH.
ENTRYPOINT ["strategy-report"]