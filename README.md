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










## Instructions to Run


STEP1: Extract Features from Video
```shell
python3 step1_extract_features.py \
    -v ./videos/session_1_cut_4to7.mp4 \
    -o ./outputs/session_1_raw
```
---

STEP2: Visualize Features
```shell
python step2_visualize_features.py \
    --features outputs/session_1_raw/ \
    --video videos/session_1_cut_4to7.mp4 \
    --output visualizations_new/session_1/
```
---

STEP3: Visualize Features
```shell
python3 step3_reconcile_ids.py \
    -f ./outputs/session_1_raw/ \
    -o ./outputs/session_1_reconciliation
```
    





2. Secondly we will download the `mobilenet0.25_Final.pth` into `class-id-clean/weights/` from this [Google Drive Link](https://drive.google.com/drive/folders/15dWsWr4dpQUf8zyuH0gi6y7fOQN5l84X?usp=sharing) 



## Run with Docker:

```docker
docker build -t classid-app .
```

```docker
docker run -it --rm \
  -v "$(pwd)/class-id-clean:/app/class-id-clean" \
  classid-app
```