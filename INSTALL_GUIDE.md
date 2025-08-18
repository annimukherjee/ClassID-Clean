# Install Guide

OpenMMLab projects are notorious for their finicky environments.

We made this document such that it becomes simple for you to use our software.

# ClassID Webcam Environment Setup Guide (Apple Silicon M1 Pro)

## Overview
This guide documents the complete setup process for the ClassID webcam analysis environment, including all errors encountered and their solutions.

## Final Working Configuration
- **Python:** 3.8
- **PyTorch:** 1.13.0
- **MMCV:** 1.6.2
- **MMDetection:** 2.28.2
- **MMPose:** 0.29.0  
- **MMTracking:** 0.14.0
- **NumPy:** 1.23.5
- **FaceNet-PyTorch:** 2.6.0

---

## Step-by-Step Installation

### Step 1: Create Base Environment
```bash
# Create conda environment with Python 3.8 (good compatibility)
conda create -n classid_webcam python=3.8 -y
conda activate classid_webcam
```

### Step 2: Install PyTorch (Pinned Version)
```bash
# Install PyTorch 1.13.0 specifically - required for Apple Silicon M1 with mmcv-full
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
```

**Why PyTorch 1.13.0?** OpenMMLab docs state: "for the mac M1 chip, only the pre-built packages on PyTorch v1.13.0 are available" for mmcv-full 1.7.x.

### Step 3: Verify PyTorch Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```
Expected output:
```
PyTorch: 1.13.0
MPS available: True  (or False, both are fine for CPU usage)
```

### Step 4: Install OpenCV (Required Dependency)
```bash
pip install opencv-python
```

### Step 5: Install mmcv-full 
```bash
# Initial attempt (this will cause issues later)
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html
```

### Step 6: Verify mmcv-full Installation
```bash
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}'); print('mmcv.Config available:', hasattr(mmcv, 'Config'))"
```

### Step 7: Install MMDetection
```bash
pip install mmdet==2.28.2
```

### Step 8: Install MMPose (First Attempt - FAILS)
```bash
pip install mmpose==0.29.0
```

**ERROR ENCOUNTERED:**
```
ERROR: Failed building wheel for xtcocotools
OSError: [Errno 66] Directory not empty: ...cython...
```

**SOLUTION - Install xtcocotools from source first:**
```bash
git clone https://github.com/jin-s13/xtcocoapi
cd xtcocoapi
python setup.py install
cd ..
rm -rf xtcocoapi
```

**Then retry mmpose:**
```bash
pip install mmpose==0.29.0
```

### Step 9: Test MMPose (FAILS with Version Conflict)
```bash
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"
```

**ERROR ENCOUNTERED:**
```
AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.3.8, <=1.7.0.
```

**SOLUTION - Downgrade mmcv-full:**
```bash
# Uninstall current mmcv-full
pip uninstall mmcv-full -y

# Install compatible version
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html
```

**Verify the fix:**
```bash
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}'); print('mmcv.Config available:', hasattr(mmcv, 'Config'))"
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"
```

### Step 10: Install MMTracking
```bash
pip install mmtrack==0.14.0
```

### Step 11: Test Core APIs (FAILS with lap package error)
```bash
python -c "
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
print('✓ All MMlab APIs imported successfully')
"
```

**ERROR ENCOUNTERED:**
```
ImportError: numpy.core.multiarray failed to import (auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy; use '<void>numpy._import_array' to disable if you are certain you don't need it).
```

**SOLUTION - Fix NumPy/lap compatibility:**
```bash
# Try installing lap from source (failed)
pip uninstall lap -y
pip install lap --no-binary lap

# Final solution - upgrade NumPy
pip uninstall numpy -y
pip install numpy==1.23.5

# Reinstall lap
pip uninstall lap -y  
pip install lap
```

**Test again:**
```bash
python -c "
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
print('✓ All MMlab APIs imported successfully')
"
```

### Step 12: Install Face Libraries (CAREFUL - Version Conflicts!)
```bash
# First attempt (BREAKS everything by upgrading PyTorch)
pip install facenet-pytorch

# This upgraded PyTorch 1.13.0 → 2.2.2, breaking mmcv!
```

**ERROR ENCOUNTERED:**
```
ImportError: dlopen(...mmcv/_ext.cpython-38-darwin.so, 0x0002): Symbol not found: __ZN2at3mps9MPSStream6commitEb
```

**SOLUTION - Revert and install carefully:**
```bash
# Uninstall problematic packages
pip uninstall facenet-pytorch torch torchvision numpy -y

# Reinstall working versions
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
pip install numpy==1.23.5

# Install facenet-pytorch WITHOUT upgrading dependencies
pip install facenet-pytorch --no-deps

# Install only needed dependencies manually
pip install requests tqdm Pillow
```

### Step 13: Final Environment Test
```bash
python -c "
# Test all core imports from your script
import os
import time
import argparse
import pickle
import cv2
import mmcv
import torch
import numpy as np
from copy import deepcopy
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from facenet_pytorch import InceptionResnetV1
from concurrent.futures import ThreadPoolExecutor

print('✓ All core dependencies imported successfully!')
print('✓ Environment is ready for your ClassID webcam script!')
print()
print('PyTorch:', torch.__version__)
print('MMCV:', mmcv.__version__)
print('NumPy:', np.__version__)
"
```

**SUCCESS OUTPUT:**
```
✓ All core dependencies imported successfully!
✓ Environment is ready for your ClassID webcam script!

PyTorch: 1.13.0
MMCV: 1.6.2
NumPy: 1.23.5
```

---

## Key Lessons Learned

### 1. **Version Compatibility is Critical**
- OpenMMLab packages have very strict version requirements
- Always check compatibility matrices before installation
- mmdet 2.x ⟷ mmpose 0.x ⟷ mmcv 1.x

### 2. **Apple Silicon Requires Specific Versions**
- PyTorch 1.13.0 is the only version with pre-built mmcv-full packages for M1
- Building from source often fails due to compilation issues

### 3. **Installation Order Matters**
- Install PyTorch first with conda (more reliable for Apple Silicon)
- Install mmcv-full before other MM packages
- Use `--no-deps` flag when installing packages that might upgrade core dependencies

### 4. **Common Error Patterns**
- **xtcocotools build failures:** Install from source first
- **NumPy/Cython errors:** Usually fixed by upgrading NumPy to compatible version
- **mmcv symbol errors:** Caused by PyTorch version mismatches

### 5. **Dependency Management Strategy**
- Pin versions for core packages (PyTorch, mmcv-full)
- Use `--no-deps` for face/auxiliary libraries
- Install missing dependencies manually to maintain control

---

## Troubleshooting Commands

### Check Current Versions
```bash
pip list | grep -E "(torch|mmcv|mmdet|mmpose|mmtrack|numpy)"
```

### Test Core Functionality
```bash
python -c "
from mmtrack.apis import inference_mot, init_model as init_tracking_model
from mmpose.apis import inference_top_down_pose_model, init_pose_model
print('✓ MMlab APIs working')
"
```

### Reset PyTorch if Corrupted
```bash
conda remove pytorch torchvision torchaudio --force -y
conda clean --all -y
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch -y
```

---

## Notes

- **TorchReID:** Skipped due to dependency complexity - can be added later if needed
- **Scipy Warning:** Ignore the NumPy version warning from scipy - everything still works
- **Custom Modules:** Don't forget to add your `FaceWrapper` and `GazeWrapper` to Python path:
  ```bash
  export PYTHONPATH="${PYTHONPATH}:/path/to/your/classid/project"
  ```

---

## Final Package List
```
torch==1.13.0
torchvision==0.14.0  
mmcv-full==1.6.2
mmdet==2.28.2
mmpose==0.29.0
mmtrack==0.14.0
numpy==1.23.5
opencv-python>=4.5.0
facenet-pytorch==2.6.0
```

**Total setup time:** ~30 minutes including troubleshooting
**Environment status:** ✅ Ready for ClassID webcam analysis script