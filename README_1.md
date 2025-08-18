# ClassID Clean

This repository contains a clean implementation of the ClassID paper.

The ClassID system has 3 phases:
1. Within Session ID Assignment
2. Individual Session Level Representation
3. Across Session ID Matching

Currently we have implemented 1. Within Session ID Assignment.

This phase contains 5 steps:

MP4 Video --> OC-SORT --> Filter Ephemeral IDs --> Perform Local Reconciliation --> Extract Features (Pose, Face BBox, Face Embeddings, Gaze Features) --> Perform Global Reconciliation.

Please refer to the paper [Section 3] ([PDF](./ClassID-Paper.pdf)) for more details.


Before we run, we need to download some large `.pth` files.


### Downloading the model Weights:

1. First we will download all the .pth files present in [link](https://drive.google.com/drive/folders/1mUtuwzOQwKuVb1XMRxDjkGUJ3bSNLPjz?usp=sharing) into a new directory we will create:

```shell
mkdir -p ./models/mmlab
```

Download all the model weights from this [Google Drive Link](https://drive.google.com/drive/folders/1mUtuwzOQwKuVb1XMRxDjkGUJ3bSNLPjz?usp=sharing)  into the `mmlab` directory that was just created...


## Run with Docker:

```docker
docker build -t classid-app:latest .
```

```docker
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  classid-app:latest
```

> **Note:** A pre-built Docker image will be available on [Docker Hub](https://hub.docker.com/) soon for easier setup.