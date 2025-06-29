# SpecDETR
This is the official repository for "SpecDETR: A transformer-based hyperspectral point object detection network", and it is also part of the open-source hyperspectral object detection toolbox [HODToolbox](https://github.com/ZhaoxuLi123/HODToolbox).




Paper link:  [ISPRS P&RS](https://www.sciencedirect.com/science/article/abs/pii/S0924271625001868) or [arXiv](https://arxiv.org/abs/2405.10148).


<br>

## Contributions

1. We verified that existing visual objec detection networks possess the capability for spatial-spectral integrated semantic representation of tiny objects. These networks can effectively detect subpixel-level objects in hyperspectral images, outperforming pixel-wise hyperspectral target detection methods.

2. We developed the first multi-class hyperspectral tiny object detectio benchmark dataset, SPOD. Based on this dataset, a comprehensive evaluation was conducted on the performance of mainstream visual object detection networks and hyperspectral target detection methods on the hyperspectral tiny object detection task.

3. We proposed SpecDETR, an innovate tiny object detection network that introduces a self-excited mechanism to enhance feature extraction and designs a streamlined yet efficient novel DETR decoder architecture tailored for tiny object characteristics. Experiments demonstrate that SpecDETR significantly outperforms existing approaches in the hyperspectral tiny object detection task.

4. We developed  an open-source hyperspectral object detection toolbox, HODToolbox, facilitating the paradigm shift from traditional pixel-level hyperspectral target detection to  hyperspectral object detection. The toolbox integrates the following core functionalities:
   - Convert traditional hyperspectral target detection datasets into object detection datasets, and use single-target prior spectra to generate large-scale training image sets for object detection networks training.
   - Train and test mainstream visual object detection networks on hyperspectral object detection datasets.
   - Quantitative evaluation and visual analysis of detection results.


<br>

## News & Updates
* June 29, 2025: We make the following updates:
    1. Open-source the simulated training sets for three public HTD datasets: Avon, SanDiego, and MUUFLGulfport.
    2. Add support for the infrared video satellite flying airplane detection dataset [IRAir](https://github.com/ZhaoxuLi123/IRAir).
    3. Provide pre-trained SpecDETR models for hyperspectral tiny object detection on three public datasets (Avon, SanDiego, MUUFLGulfport) and single-frame infrared tiny object detection on IRAir dataset.
    4. Released the companion toolbox [HODToolbox](https://github.com/ZhaoxuLi123/HODToolbox) 

* May 08, 2025: We are pleased to announce that our work SpecDETR has been accepted by ISPRS Journal of Photogrammetry and Remote Sensing!



<br>

## Abstract

Hyperspectral target detection (HTD) aims to identify specific materials based on spectral information in hyperspectral imagery and can detect extremely small-sized objects, some of which occupy a smaller than one-pixel area. However, existing HTD methods are developed based on per-pixel binary classification, neglecting the three-dimensional cube structure of hyperspectral images (HSIs) that integrates both spatial and spectral dimensions. The synergistic existence of spatial and spectral features in HSIs enable objects to simultaneously exhibit both, yet the per-pixel HTD framework limits the joint expression of these features. In this paper, we rethink HTD from the perspective of spatial–spectral synergistic representation and propose hyperspectral point object detection as an innovative task framework. We introduce SpecDETR, the first specialized network for hyperspectral multi-class point object detection, which eliminates dependence on pre-trained backbone networks commonly required by vision-based object detectors. SpecDETR uses a multi-layer Transformer encoder with self-excited subpixel-scale attention modules to directly extract deep spatial–spectral joint features from hyperspectral cubes. During feature extraction, we introduce a self-excited mechanism to enhance object features through self-excited amplification, thereby accelerating network convergence. Additionally, SpecDETR regards point object detection as a one-to-many set prediction problem, thereby achieving a concise and efficient DETR decoder that surpasses the state-of-the-art (SOTA) DETR decoder. We develop a simulated hyperSpectral Point Object Detection benchmark termed SPOD, and for the first time, evaluate and compare the performance of visual object detection networks and HTD methods on hyperspectral point object detection. Extensive experiments demonstrate that our proposed SpecDETR outperforms SOTA visual object detection networks and HTD methods. 





<br>


## Installation

The project is built upon mmdetection 3.0.0 and runs on the Ubuntu system.

1. Create a new conda environment and activate the environment. Requires Python>=3.7.

2. Install Pytorch. Requires torch>=1.8.

3. Install mmengine and mmcv.
    ```bash
    pip install mmengine==0.7.3
    pip install mmcv==2.0.0
    ```
   
4. Clone this repository:
    ```bash
    SpecDETR_ROOT=/path/to/clone/SpecDETR
    git clone https://github.com/ZhaoxuLi123/SpecDETR $SpecDETR_ROOT
    ```
   
5. Compile and install mmdet. If mmdet>=3.0.0 is already installed in the environment, skip this step.
    ```bash
    cd $SpecDETR_ROOT
    pip install -v -e .
    ```

<br>

## Dataset Preparation


### SPOD Dataset


1. Download SPOD_30b_8c.zip from [Baidu Drive](https://pan.baidu.com/s/1fySVhp4w2coz1vwvB6aSgw?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/Ea8D-QY1zoxKq8a1xCj0XXoB4dNWd-M2BM3FvYV042JHXw).

2. Unzip SPOD_30b_8c.zip locally.

3. Update the dataset configuration files [configs/\_base_/datasetshsi_detection.py](configs/_base_/datasets/hsi_detection.py) and [configs/VisualObjectDetectionNetwork/\_base_/datasets/hsi_detection4x.py](configs/VisualObjectDetectionNetwork/_base_/datasets/hsi_detection4x.py):

      ```python
        data_root = 'Dataset_Root/SPOD_30b_8c/'  # ← Update to the local path
      ```

### Avon Dataset, SanDiego Dataset, and MUUFLGulfport Dataset 

1. Download  dataset zip files:
   - **Avon Dataset**: [Baidu Drive](https://pan.baidu.com/s/13yIPxUulRAa0-s_O_eFL8w?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EXI7DG5slNdCkAla1QqNUagBA2V_cXgw_Oj8p5tHijttAg)  

   - **SanDiego Dataset**: [Baidu Drive](https://pan.baidu.com/s/1bKUFdZC0GQYDUSPRh5QBpw?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EaZqpRVz_nxPgG8ufedM0U0BslexIeE138_RGYXOcMgjpw)  

   - **MUUFLGulfport Dataset**: [Baidu Drive](https://pan.baidu.com/s/1xWA45V92eGEs29tJvNl8AA?pwd=2789) (key: 2789) or [OneDrive](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EfsT0_HjSQJLj5c-Hy777MUBS9a0wBTYQlktLtm5rz4E5w)

2. Unzip dataset zip files locally.

3. Update the dataset configuration files [configs/\_base_/datasetshsi_detection.py](configs/_base_/datasets/hsi_detection.py) and [configs/VisualObjectDetectionNetwork/\_base_/datasets/hsi_detection4x.py](configs/VisualObjectDetectionNetwork/_base_/datasets/hsi_detection4x.py):

      ```python
        data_root = 'Dataset_path/'  # ← Update to the local path of the dataset.
      ```

### Other HTD Dataset

We provide conversion scripts from HTD Dataset to HOD Dataset in [HODToolbox](https://github.com/ZhaoxuLi123/HODToolbox).  

For details on adding new HOD datasets, please refer to the `Training and Inference of the HOD Task` section in [HODToolbox](https://github.com/ZhaoxuLi123/HODToolbox).  


### IRAir Dataset

We will release the IRAir dataset soon. Please stay tuned on the project website: [IRAir](https://github.com/ZhaoxuLi123/IRAir)  

The IRAir dataset configuration file is located at:  
[configs/\_base_/datasets/irair_real_label.py](configs/_base_/datasets/irair_real_label.py)  

<br>

## Model Zoo

We provide pre-trained SpecDETR models for SPOD Dataset, Avon Dataset, SanDiego Dataset, MUUFLGulfport Dataset, and IRAir Dataset.


|    Method     |           Dataset           |                             Configuration                                             |                                   Baidu Drive                                   |         OneDrive       |
|:-------------:|:---------------------------:|:-------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:---------------------:|
|   SpecDETR	   |       SPOD Dataset 	        |                 [Link](configs/specdetr/SpecDETR_SPOD_100e.py) 	                 | [Link](https://pan.baidu.com/s/1v6pyjOPdEfwu7peqCjqfpw?pwd=2789) (key: 2789)  	 | [Link](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EXgsOVC5S9NOuTLMsHi2sCwBOo_lAMv7FZ1lcuawABupkw) |
|   SpecDETR    |        Avon Dataset	        |                  [Link](configs/specdetr/SpecDETR_Avon_36e.py)                  |                                   [Link](https://pan.baidu.com/s/1GCClB-Y-4Tpn_wWGzu3zbA?pwd=2789) (key: 2789)  	                                    |  [Link](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/ET6gANP41KhHk3DUlCLhMWgBFkHrMn4V_7FireaoyAXJ3g)  |
|   SpecDETR    |  SanDiego Dataset	 	        |                  [Link](configs/specdetr/SpecDETR_Sandiego_12e.py)                  |                                    [Link](https://pan.baidu.com/s/1v1kdryPt910A5NbkXf3kqA?pwd=2789) (key: 2789) 	                                     |  [Link](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/Een05DyaF0VCpf2kBNVNH3MBY3ebJGBtEtnhVnyaTqM_Sw)  |
|   SpecDETR    | MUUFLGulfport Dataset	    	 |                 [Link](configs/specdetr/SpecDETR_MUUFLGulfport_24e.py) 	                 |                               [Link](https://pan.baidu.com/s/1UdOHQiAKrc25TgmXi349Hw?pwd=2789) (key: 2789) 	                                     |  [Link](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/ERoonO23CZtEuB71Ls3dhWkBOm5r5DEra9yuTtAXOLFfgw)  |
| SpecDETR	     |     IRAir Dataset	    	     |                  [Link](configs/specdetr/SpecDETR_IRAir_12e.py)                  |                                     [Link](https://pan.baidu.com/s/14-JcTfu1H6biz5sZCdC_KQ?pwd=2789) (key: 2789) 	                                     |  [Link](https://dh7n-my.sharepoint.com/:u:/g/personal/vv_s3_tm9_site/EVNJWi7wtrZJpohz3g37pFsBgSMz2sjVxnbvqm2SgzD_7g)  |


<br>

## Model Training

### SpecDETR 

To train SpecDETR on the SPOD dataset, execute either of the following commands:

```bash
python train.py --dataset SPOD
```
or
```bash
python train.py --config ./configs/specdetr/SpecDETR_SPOD_100e.py --work-dir ./work_dirs/SpecDETR/SPOD/
```

Note: Although we have fixed all random seeds, there may still be slight differences in AP performance each time you run. This difference originates from the underlying mechanism of CUDA.

### Other Model 

We provide configuration files for existing visual object detection networks on the SPOD dataset in the [configs/VisualObjectDetectionNetwork](configs/VisualObjectDetectionNetwork) directory. Execute the following command to train these models:

```bash
python train.py --config ./configs/VisualObjectDetectionNetwork/dino-5scale_swin-l-100e_hsi4x.py --work-dir ./work_dirs/dino-5scale_swin-l/SPOD/
```

<br>

## Model Evaluation  
Use SPOD dataset as example. 

1. **Model Inference**  
   Run the following command to obtain SpecDETR's inference results on the SPOD test set:  
   ```bash  
   python test.py --dataset SPOD  
   ```  
   or  
   ```bash  
   python test.py --config ./configs/specdetr/SpecDETR_SPOD_100e.py \  
                 --work-dir ./work_dirs/SpecDETR/SPOD/ \  
                 --checkpoint ./work_dirs/SpecDETR/SpecDETR_SPOD_100e.pth \  
                 --out ./work_dirs/SpecDETR/SPOD/SpecDETR.pkl  
   ```  

2. **Detection Accuracy Evaluation**  
   Please refer to the `Quantitative Evaluation of Results` section in [HODToolbox](https://github.com/ZhaoxuLi123/HODToolbox).  

3. **Inference Speed Evaluation**  
   Execute the following command to evaluate SpecDETR's inference speed on the SPOD test set:  
   ```bash  
   python benchmark.py --config ./configs/specdetr/SpecDETR_SPOD_100e.py \  
                      --checkpoint ./work_dirs/SpecDETR/SpecDETR_SPOD_100e.pth  
   ```  

4. **FLOPs Calculation**  
   Run the following command to compute FLOPs:  
   ```bash  
   python get_flops.py --config ./configs/specdetr/SpecDETR_SPOD_100e.py  
   ```
<br>

## Benchmark 


### Partial Quantitative Results of Visual Object Detection Networks on SPOD Dataset


|       Method       |   Backbone       | Image Size |       mAP50:95        |        mAP25        |        mAP50        |          mAP75        |      FLOPs       |        Params    |
|:------------------:|:------------:|:----------:|:---------------------:|:-------------------:|:-------------------:|:---------------------:|:---------------------:|:---------------:|
|  **Faster R-CNN**  | **ResNet50** |     x4     |     0.197    |   0.377    |    0.374    |     0.179     |     68.8G     | 41.5M   |
|  **Faster R-CNN**  | **RegNetX** |     x4     |     0.227    |   0.379   |    0.378    |    0.242    |     57.7G     | 31.6M   |
|  **Faster R-CNN**  | **ResNeSt50** |     x4     |     0.246    |   0.316    |    0.316    |     0.277     |     185.1G     | 44.6M   |
|  **Faster R-CNN**  | **ResNeXt101** |     x4     |     0.220    |   0.368   |    0.366   |     0.231     |     128.4G     | 99.4M   |
|  **Faster R-CNN**  | **HRNet** |     x4     |     0.320    |   0.404    |    0.402   |     0.345     |     104.4G     | 63.2M   |
|      **TOOD**      | **ResNeXt101** |     x4     |     0.304    |   0.464    |    0.440    |     0.303    |     114.3G     | 97.7M   |
| **CentripetalNet** | **HourglassNet104** |     x4     |    0.695    |   0.829    |   0.805    |     0.673     |     501.3G     | 205.9M   |
|   **CornerNet**    | **HourglassNet104** |     x4     |     0.626    |   0.736   |    0.712    |     0.609     |    462.6G     | 201.1M   |
|   **RepPoints**    | **ResNet50** |     x4     |     0.207   |   0.691   |    0.572   |     0.074     |     54.1G     | 36.9M   | |
|   **RepPoints**    | **ResNeXt101** |     x4     |   0.485    |   0.806    |    0.790   |     0.540     |     75.0G     | 58.1M   |
|   **RetinaNet**    | **EfficientNet** |     x4     |    0.462   |  0.836    |    0.811   |     0.466     |     36.1G     | 18.5M   |
|   **RetinaNet**    | **PVTv2-B3** |     x4     |   0.426    |  0.757   |    0.734    |    0.442     |     71.3G     | 52.4M   |
| **DeformableDETR** | **ResNet50** |     x4     |     0.231    |   0.692    |    0.560    |     0.147     |     58.7G     | 41.2M   |
|      **DINO**      | **ResNet50** |     x4     |    0.168   |   0.491   |    0.418   |    0.097    |    86.3G     | 47.6M   |
|      **DINO**      | **Swin-L** |     x4     |    0.757    |   0.852   |    0.842    |    0.764     |     203.9G     | 218.3M   |
|    **SpecDETR**    | -- |     x1     |     **0.856**    |  **0.938**    |    **0.930**   |     **0.863**   |     139.7G     | 16.1M   |



### Partial Quantitative Results of HTD Methods on SPOD Dataset


|  Method    |     mAP50:95     |   mAP25   |        mAP50        |          mAP75        |
|:----------:|:----------------:|:---------:|:-------------------:|:---------------------:|
| **ASD**	 |     0.182 	      |  0.286 	  | 0.260 	 | 0.182  |
| **CEM** |     0.040 	      |  0.122 	  | 0.075 	 | 0.035 |
| **CRBBH**	 |    0.036 	 	     |  0.129 	  | 0.083 	 | 0.028 |
| **CSRBBH**	 |   0.034 	    	   |  0.116 	  | 0.076 	 | 0.028 |
| **HSS**	 |     0.073 	      |  0.303 	  | 0.179 	 | 0.058 |
| **IRN**	 | 0.000 	       	  |  0.002 	  | 0.001 	 | 0.000 |
| **KMSD**	 |   0.108 	    	   |  0.285 	  | 0.207 	 | 0.095 |
|  **KOSP**	 |     0.017 	      |  0.083 	  | 0.044 	 | 0.014 |
|  **KSMF**	 | 0.003 	       	  |  0.015 	  | 0.009 	 | 0.002 |
|  **KTCIMF**	 | 0.001 	       	  |  0.008 	  | 0.002 	 | 0.000 |
|  **LSSA**	 |  0.041 	      	  |  0.093 	  | 0.071 	 | 0.037 |
|  **MSD**	 | 0.248 	       	  |  0.521 	  | 0.402 	 | 0.228 |
|  **OSP**	 | 0.031 	       	  |  0.108 	  | 0.063 	 | 0.027 |
|  **SMF**	 |  0.003 	     	   |  0.016 	  | 0.008 	 | 0.002 |
|  **SRBBH**	 | 0.019 	       	  |  0.092 	  | 0.051 	 | 0.013 | 
|  **SRBBH_PBD**	 | 0.013 	       	  |  0.088 	  | 0.038 	 | 0.007 |
|  **TCIMF**	 | 0.009 	       	  |  0.061 	  | 0.025 	 | 0.007 |
|  **TSTTD**	 | 0.044 	        	 |  0.057 	  | 0.055 	 | 0.043 |
|  **SpecDETR**  	 |    **0.856**     | **0.938** 	 | **0.930** 	 | **0.863** |



<br>

## Citation

If the work or the code is helpful, please cite the paper:
```
@article{li2025specdetr,
  title={SpecDETR: A transformer-based hyperspectral point object detection network},
  author={Li, Zhaoxu and An, Wei and Guo, Gaowei and Wang, Longguang and Wang, Yingqian and Lin, Zaiping},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={226},
  pages={221--246},
  year={2025},
  publisher={Elsevier}
}
```

<br>

## Contact

For further questions or details, please directly reach out to lizhaoxu@nudt.edu.cn.