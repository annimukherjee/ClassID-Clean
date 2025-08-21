# ClassID-Clean

This repository contains a modular Python implementation of the research paper, "[ClassID: Enabling Student Behavior Attribution from Ambient Classroom Sensing Systems](ClassID-Paper.pdf)".

The OpenMMLab libraries used in this research can be challenging to configure. This project aims to provide a straightforward path to running the core pipeline by using Docker to handle the entire environment setup.

## Pipeline Overview

This project currently implements the first phase of the ClassID system: **Within-Session ID Assignment**.

The pipeline takes a classroom video as input and processes it through five stages to produce a stable, anonymous ID for each person in the session:

1.  **Initial Tracking:** Uses the OC-SORT algorithm to generate raw track IDs.
2.  **Ephemeral Filtering:** Removes noisy, short-lived tracks.
3.  **Local Reconciliation:** Merges tracks separated by short time gaps and occlusions.
4.  **Feature Extraction:** Gathers pose, face, gaze, and facial embedding data for each track.
5.  **Global Reconciliation:** Uses facial features and location to merge tracks separated by long time gaps (e.g., a student leaving and returning).

The final output is a clean dataset ready for further analysis.

---

## How to Run

This project is designed to be run inside a Docker container. You will not need to install any Python packages on your local machine.

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) must be installed and running.
-   `git` is needed to clone the repository.

### Step 1: Clone the Repository

First, get the project files onto your local machine.

```bash
git clone https://github.com/annimukherjee/ClassID-Clean.git
cd ClassID-Clean
```

### Step 2: Download Model Weights

The build process requires several pre-trained model files. These are large, so they are not included in the repository.

1. Create the necessary directory for the models
```bash
mkdir -p ./models/mmlab
```
2. Download the model files from this [Google Drive Link](https://drive.google.com/drive/folders/1mUtuwzOQwKuVb1XMRxDjkGUJ3bSNLPjz).
3. Place the downloaded .pth files into the `./models/mmlab/` directory.



### Step 3: Run the Pipeline

This project is designed to be run with Docker. The easiest way is to use the pre-built image from Docker Hub.

**Option A: Run with the Pre-built Docker Image (Recommended)**

1. Pull the Docker Image:
Download the ready-to-use image from Docker Hub with this command:
```shell
docker pull animukh/classid-pipeline-cpu
```

2. Run the Pipeline:
Place your video file inside the `input/` directory and ensure it is named `video.mp4`. Then, run the container with the following command:

```shell
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  animukh/classid-pipeline-cpu
```

The `-v` flags create a shared folder between your computer and the container, allowing it to read your video and save the results back to your `output/` folder.


**Option B: Build the Image from Scratch**

If you need to modify the project or build the image locally, you can follow these steps.


1. Build the Docker Image:
This command reads the Dockerfile and builds the complete application. This will take a significant amount of time (15-30 minutes) the first time you run it.

```
# Run this from the project's root directory
docker build -t classid-clean .
```

2. Run the Pipeline:
Once the build is complete, run the locally-built container:

```
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  classid-clean
```

### Understanding the Output

After the run is complete, your `output/` folder will contain:
1. Data Files (.pkl): The processed data from each step of the pipeline. The final result is `5_globally_reconciled.pkl`.
2. Visualizations (.png): Graphs showing the lifespan of track IDs at different stages, which helps in visualizing how the data is cleaned and reconciled.
3. Annotated Videos (.mp4): For each processing stage, a video is generated with bounding boxes and track IDs drawn on each frame. This allows you to visually inspect the results of each step, from the raw output of `video_1_oc_sort_ids.mp4` to the final `video_5_global_reconciliation.mp4`.



### Project Status
1. Within-Session ID Assignment (Done)
2. Individual Session-Level Representation
3. Across-Session ID Matching


### Acknowledgments
This project is an implementation based on the excellent work of the original authors. Full credit for the methodology goes to them. Please cite their paper if you use this work:


```
Patidar, P., Ngoon, T. J., Zimmerman, J., Ogan, A., & Agarwal, Y. (2024). ClassID: Enabling Student Behavior Attribution from Ambient Classroom Sensing Systems. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 8, 2, Article 55 (June 2024), 28 pages. https://doi.org/10.1145/3659586
```