# SpecDETR
This is the official repository for ["SpecDETR: A Transformer-based Hyperspectral Point Object Detection Network"](https://arxiv.org/abs/2405.10148).

SpecDETR is the first multi-class hyperspectral point object detection work, capable of handling subpixel-sized object with extremely low target abundance.

In the point object detection task, SpecDETR outperforms DINO.

In addition, it can also be applied to infrared small target detection and other weak and small object detection tasks.

## Installation

The code is built upon mmdetection 3.0.0 and runs on the Ubuntu system.

1. Create a new conda environment and activate the environment. Requires Python>=3.7.

2. Install Pytorch. Requires torch>=1.8.

3. Install mmengine and mmcv.
    ```
    pip install mmengine==0.7.3
    pip install mmcv==2.0.0
    ```
   
4. Clone this repository:
    ```
    SpecDETR_ROOT=/path/to/clone/SpecDETR
    git clone https://github.com/ZhaoxuLi123/SpecDETR $SpecDETR_ROOT
    ```
   
5. Compile and install mmdet. If mmdet>=3.0.0 is already installed in the environment, skip this step.
    ```
    cd SpecDETR_ROOT
    pip install -v -e .
    ```
   
## SPOD Dataset Preparation

1. Download SPOD_30b_8c.zip from [Baidu Netdisk](https://pan.baidu.com/s/1vw23KWPSus2Yuj-CA1URnw?pwd=1234) (code: 1234) or [Google Drive](https://drive.google.com/file/d/1wfoLkfZOxxEtuDyDSWCnAq-N8HZVKdQ1/view?usp=drive_link).

2. Unzip SPOD_30b_8c.zip to the specified path.
    ```
    SPOD_ROOT=/path/to/unzip/SPOD_dataset
    unzip -d SPOD_ROOT SPOD_30b_8c.zip
    ```
   
3. Update the dataset configuration file ./configs/_base_/datasets/hsi_detection.py
    ```
    Line3 data_root = 'SPOD_ROOT/SPOD_30b_8c/'
    ```
   
## Benchmark Evaluation and Training

### Pre-trained Weights Preparation

1. Download the pre-trained weights file SpecDETR_100e.pth from [Baidu Netdisk](https://pan.baidu.com/s/12-33-sCQWcMYUy5QU7rVNQ?pwd=1234) (code: 1234) or [Google Drive](https://drive.google.com/file/d/1h6_MzTb_jQ-7I09x2qfg4vci4hA50hhY/view?usp=drive_link).

2. Place the file SpecDETR_100e.pth under ./work_dirs/SpecDETR/

### Evaluation

1. Obtain detection results and evaluate AP.
    ```
    python test.py
    ```
   
2. Evaluate inference speed FPS.
    ```
    python train.py
    ```
   
3. Evaluate flops.
    ```
    python benchmark.py
    ```
   
4. Retrain the network from scratch.
    ```
    python train.py
    ```
Note: Although we have fixed all random seeds, there may still be slight differences in AP performance each time you run. This difference originates from the underlying mechanism of the mmdetection framework.

## Citation

If the work or the code is helpful, please cite the paper:
```
@article{li2024specdetr,
  title={SpecDETR: A Transformer-based Hyperspectral Point Object Detection Network},
  author={Li, Zhaoxu and An, Wei and Guo, Gaowei and Wang, Longguang and Wang, Yingqian and Lin, Zaiping},
  journal={arXiv preprint arXiv:2405.10148},
  year={2024}
}
```

## Contact

For further questions or details, please directly reach out to lizhaoxu@nudt.edu.cn.