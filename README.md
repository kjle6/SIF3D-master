# SIF3D: Multimodal Sense-Informed Forecasting of 3D Human Motions (CVPR 2024)

## Introduction

This is the official repo of our paper [SIF3D: Multimodal Sense-Informed Forecasting of 3D Human Motions].

For more information, please visit our [project page](https://sites.google.com/view/cvpr2024sif3d).

## Setup
The following setup borrows the setting of [GIMO](https://github.com/y-zheng18/GIMO).

To setup the environment, firstly install the packages in requirements.txt:

```
pip install -r requirements.txt
```

Install PointNet++ as described [here](https://github.com/daerduoCarey/o2oafford/tree/main/exps) :

```
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
# [IMPORTANT] Also, you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
pip install -r requirements.txt
pip install -e .
```

Download and install [Vposer](https://github.com/nghorbani/human_body_prior), [SMPL-X](https://github.com/vchoutas/smplx)


## Dataset
SIF3D applies dataset processed on our own. However, in order to comply with dataset confidentiality rules, we can not release the processed version of the dataset.
Please follow the instructions of the official repo of [GIMO](https://github.com/y-zheng18/GIMO?tab=readme-ov-file#dataset) to download the raw dataset.
After downloading and unzipping, you will get a folder like this:
```
--data_root
     |--bedroom0122
           |--2022-01-21-194925
                 |--eye_pc
                 |--PV
                 |--smplx_local
                 |--transform_info.json
                 ...
           |--2022-01-21-195107
           ...
     |--bedroom0123
     |--bedroom0210
     |--classroom0219
     ...
```

Our code will automatically complete the pre-process procedure during first run. Make sure to change the **dataroot** before running. After first run, the processed data would be in the same place of the raw dataset, and you will find your dataset folder like this:
```
--data_root
      |--SLICES_8s
            |--train
                 |--gazes.pth
                 |--joints_input.pth
                 |--joints_label.pth
                 |--poses_input.pth
                 |--poses_label.pth
                 |--scene_points_<sample_points>.pth
            |--test
                 |--gazes.pth
                 |--joints_input.pth
                 |--joints_label.pth
                 |--poses_input.pth
                 |--poses_label.pth
                 |--scene_points_<sample_points>.pth
     |--bedroom0122
     |--bedroom0123
     |--bedroom0210
     |--classroom0219
     ...
```

## Quickstart
### Evaluating
Simply run:
```
bash scripts/eval.sh
```

(Optional) Download our [pre-trained SIF3D weight](https://drive.google.com/file/d/10e4VTTX4zZlnFdTiNDHle9FR208i69mC/view?usp=sharing), and don't forget to change the **load_model_dir** before runing scripts/eval.sh.

### Training
Simply run:
```
bash scripts/train.sh
```
You can change the checkpoint and log saving directory by changing the **save_path** argument in scripts/train.sh.

### Metrics
The **loss_trans**, **loss_des_trans**, **mpjpe** and **des_mpjpe** corresponding to **Traj-path**, **Traj-dest**, **MPJPE-path** and **MPJPE-dest** in the paper, respectively.


## Citation
If you find this repo useful for your research, please consider citing:
```
@inproceedings{lou2024multimodal,
  title={Multimodal Sense-Informed Prediction of 3D Human Motions},
  author={Lou, Zhenyu and Cui, Qiongjie and Wang, Haofan and Tang, Xu and Zhou, Hong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
