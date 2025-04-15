#!/bin/bash

# Check if version number is provided
if [ -z "$1" ]; then
    echo "Please provide a version number (e.g., 1.0.0)"
    exit 1
fi

VERSION=$1
IMAGE_NAME="syzygianinfern0/stormbase"

# Build the Docker image with version tag
echo "Building Docker image with version $VERSION..."
docker build -f Dockerfile.stormbase -t $IMAGE_NAME:$VERSION .

# Tag as latest
echo "Tagging as latest..."
docker tag $IMAGE_NAME:$VERSION $IMAGE_NAME:latest

# Push both tags to Docker Hub
echo "Pushing to Docker Hub..."
docker push $IMAGE_NAME:$VERSION
docker push $IMAGE_NAME:latest

echo "Done! Image $IMAGE_NAME:$VERSION has been built and pushed to Docker Hub"
