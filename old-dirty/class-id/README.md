# ClassID: Enabling Student Behavior Attribution from Ambient Classroom Sensing Systems

[[paper (IMWUT 2024)](https://dl.acm.org/doi/pdf/10.1145/3659586)]


## Abstract
Ambient classroom sensing systems offer a scalable and non-intrusive way to find connections between instructor actions and student behaviors, creating data that can improve teaching and learning. While these systems effectively provide aggregate data, getting reliable individual student-level information is difficult due to occlusion or movements. Individual data can help in understanding equitable student participation, but it requires identifiable data or individual instrumentation. We propose ClassID, a data attribution method for within a class session and across multiple sessions of a course without these constraints. For within-session, our approach assigns unique identifiers to 98% of students with 95% accuracy. It significantly reduces multiple ID assignments compared to the baseline approach (3 vs. 167) based on our testing on data from 15 classroom sessions. For across-session attributions, our approach, combined with student attendance, shows higher precision than the state-of-the-art approach (85% vs. 44%) on three courses. Finally, we present a set of four use cases to demonstrate how individual behavior attribution can enable a rich set of learning analytics, which is not possible with aggregate data alone.



### Environment Setup:

### a. Clone (or Fork!) this repository

```shell
foo@bar:~$ git clone https://github.com/edusense/classid.git
```

### b. Create a virtual environment, install python packages and openmm

We recommend using conda. Tested on `Ubuntu 22.04`, with `python 3.8`.

```shell
foo@bar:~$ conda create --name classid python=3.8 -y
foo@bar:~$ conda activate classid
foo@bar:~$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
foo@bar:~$ pip install mmcv==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13.0/index.html
foo@bar:~$ pip install mmpose==0.29.0
foo@bar:~$ pip install -U openmim
foo@bar:~$ mim install mmengine
foo@bar:~$ mim install mmdet==2.28.2
foo@bar:~$ pip install -e git+https://github.com/open-mmlab/mmaction2.git@0c6182f8007ae78b512d9dd7320ca76cb1cfd938#egg=mmaction2
```

```shell
conda create --name classid python=3.8 -y
conda activate classid
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install -U openmim
mim install -U mmcv-full==1.7.0 mmdet==2.28.2  mmengine==0.7.2 mmpose==0.29.0
cd generate_av_labels/mmaction2
pip install -e .
cd ../
pip install -r requirements.txt
# download config files for mmdetection 
mim download mmdet --config faster_rcnn_r50_fpn_2x_coco.py --dest otc_models/model_ckpts
```

Additional Instructions for running individual module coming soon.

## Reference

For more details, contact [prasoonpatidar@cmu.edu](prasoonpatidar@cmu.edu).

### If you find this module useful in your research, please consider cite:
    
```bibtex
@article{10.1145/3659586,
author = {Patidar, Prasoon and Ngoon, Tricia J. and Zimmerman, John and Ogan, Amy and Agarwal, Yuvraj},
title = {ClassID: Enabling Student Behavior Attribution from Ambient Classroom Sensing Systems},
year = {2024},
issue_date = {May 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {8},
number = {2},
url = {https://doi.org/10.1145/3659586},
doi = {10.1145/3659586},
abstract = {Ambient classroom sensing systems offer a scalable and non-intrusive way to find connections between instructor actions and student behaviors, creating data that can improve teaching and learning. While these systems effectively provide aggregate data, getting reliable individual student-level information is difficult due to occlusion or movements. Individual data can help in understanding equitable student participation, but it requires identifiable data or individual instrumentation. We propose ClassID, a data attribution method for within a class session and across multiple sessions of a course without these constraints. For within-session, our approach assigns unique identifiers to 98\% of students with 95\% accuracy. It significantly reduces multiple ID assignments compared to the baseline approach (3 vs. 167) based on our testing on data from 15 classroom sessions. For across-session attributions, our approach, combined with student attendance, shows higher precision than the state-of-the-art approach (85\% vs. 44\%) on three courses. Finally, we present a set of four use cases to demonstrate how individual behavior attribution can enable a rich set of learning analytics, which is not possible with aggregate data alone.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = may,
articleno = {55},
numpages = {28},
keywords = {Behavior Attribution, Classroom Sensing, Computer Vision, Pedagogy}
}
```
