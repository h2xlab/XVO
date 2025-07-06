[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]  

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/  
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg  


# XVO: Generalized Visual Odometry via Cross-Modal Self-Training
This repository contains the code that accompanies our ICCV 2023 paper [XVO: Generalized Visual Odometry via Cross-Modal Self-Training](https://genxvo.github.io/resources/XVO.pdf). Please find our [project page](https://genxvo.github.io/) for more details. 

<p>
    <img alt="Example 1" src="images/generalized_vo.png" class="fit"/>
</p>  

### Overview  
We propose XVO, a semi-supervised learning method for training generalized monocular Visual Odometry (VO) models with robust off-the-self operation across diverse datasets and settings. Our XVO can efficiently learn to recover relative pose with real-world scale from visual scene semantics, i.e., without relying on any known camera parameters. Our key contribution is twofold. First, we empirically demonstrate the benefits of semi-supervised training for learning a general-purpose direct VO regression network. Second, we demonstrate multi-modal supervision, including segmentation, flow, depth, and audio auxiliary prediction tasks, to facilitate generalized representations for the VO task.   

### Dataset
We use KITTI, Argoverse 2 and nuScenes dataset along with in-the-wild YouTube videos (available soon). Please find their websites for dataset setup.  
| Dataset  | Download Link |
| ------------- | ------------- |
| KITTI  | [Link](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)  |
| Argoverse 2  | [Link](https://www.argoverse.org/av2.html#download-link)  |
| nuScenes  | [Link](https://www.nuscenes.org/nuscenes#download)  |
| YouTube  | Approximately 30 hours of driving footage were selected from videos published on the YouTube channel youtube.com/@jutah, featuring a diverse range of driving scenarios. A more comprehensive list of driving videos from YouTube can be found [here](https://docs.google.com/spreadsheets/d/1bHWWP_VXeEe5UzIG-QgKFBdH7mNlSC4GFSJkEhFnt2I/edit?gid=0#gid=0).|

### Environment Requirements and Installation
```bash
# create a new environment
conda create --name xvo python=3.9
conda activate xvo

# install pytorch1.13.0
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```
Our environment also requires pytorch3d, and please refer to [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for installation guidlines.

## Training 
Coming soon!

## Evaluation
```
python3 test.py
cd vo-eval-tool
python3 eval_odom.py
```
VO evaluation tool is revised from https://github.com/Huangying-Zhan/kitti-odom-eval.

## Result   
We find that incorporating audio and segmentation tasks as part of the semi-supervised learning process significantly improves ego-pose estimation on KITTI.   
<p>
    <img width=750 class="center" alt="Demo 1" src="images/qua_result.png"/ >  
</p>   

## Contact  
Please don't hesitate to contact us if you have any remarks or questions at leilai@bu.edu or sgzk@bu.edu. 

## License  
Our work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].  

## ToDos  
- [X] Test code release
- [ ] Training code release
- [ ] Readme Update
