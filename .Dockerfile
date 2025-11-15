# Stage 1: Use a standard, lightweight Python base image
FROM python:3.12-slim

# Stage 2: System Setup
# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# Set the working directory in the container
WORKDIR /app

# Stage 3: Application Setup
# Copy only the dependency definition first to leverage Docker's layer caching
COPY pyproject.toml .

# Install the project and its dependencies. This reads pyproject.toml.
RUN pip install .

# Copy the application source code and assets
COPY src/ ./src
COPY assets/ ./assets

# Stage 4: Execution
# Set the entrypoint to our installed script, making the container executable
ENTRYPOINT ["strategy-report"]

# Define the default command to run
CMD ["run"]