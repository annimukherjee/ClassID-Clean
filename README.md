# ClassID Clean


This repository contains a clean implementation of the ClassID paper.

I first try to get `ClassID/legacy_code/trackling_pipeline_singlethread.py` to run locally on my MBP 32GB RAM machine. Then I will also try to get `ClassID/legacy_code/reid_added_pipeline_singlethread.py` to run on my machine locally.


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
    

### Downloading the model Weights:

1. First we will download all the .pth files present in [link](https://drive.google.com/drive/folders/1mUtuwzOQwKuVb1XMRxDjkGUJ3bSNLPjz?usp=sharing) into a new directory we will create:
```shell
mkdir -p ./class-id-clean/models/mmlab
```

Download all the model weights from https://drive.google.com/drive/folders/1mUtuwzOQwKuVb1XMRxDjkGUJ3bSNLPjz?usp=sharing into the `mmlab` directory that was just created...



2. Secondly we will download the `mobilenet0.25_Final.pth` into `class-id-clean/weights/` from https://drive.google.com/drive/folders/15dWsWr4dpQUf8zyuH0gi6y7fOQN5l84X?usp=sharing