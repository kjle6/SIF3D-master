import os
import json
import trimesh
import pickle
import smplx
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from human_body_prior.tools.model_loader import load_vposer
from utils.vis_utils import latent_to_joints
from pointnet2_ops import pointnet2_utils
from scipy.spatial.transform import Rotation


class EgoEvalDataset(data.Dataset):
    def __init__(self, config, train=False):
        folder = "SLICES_888s"
        split = "train" if train else "test"
        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.fps = config.fps
        self.sequences_path_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))
        self.parse_data_info()

        # If this is the first time, preprocess raw dataset, this would take ~1min.
        os.makedirs(os.path.join(self.dataroot, folder), exist_ok=True)
        if not os.path.exists(os.path.join(self.dataroot, folder, split)):
            print(f"This is the first time to build dataset, processing {split} dataset...")
            self.prepocess_dataset(folder, split)

        self.gazes = torch.load(os.path.join(self.dataroot, folder, split, "gazes.pth"))
        self.poses_input = torch.load(os.path.join(self.dataroot, folder, split, "poses_input.pth"))
        self.poses_label = torch.load(os.path.join(self.dataroot, folder, split, "poses_label.pth"))
        self.joints_input = torch.load(os.path.join(self.dataroot, folder, split, "joints_input.pth"))
        self.joints_label = torch.load(os.path.join(self.dataroot, folder, split, "joints_label.pth"))
        self.scenes = torch.load(os.path.join(self.dataroot, folder, split,
                                              "scene_points_" + str(config.sample_points) + ".pth"))

    def __getitem__(self, index):
        scene = self.scenes_path_list[index]
        seq = self.sequences_path_list[index]

        gazes = self.gazes[index]
        poses_input = self.poses_input[index]
        poses_label = self.poses_label[index]
        joints_input = self.joints_input[index]
        joints_label = self.joints_label[index]
        scene_points = self.scenes[index]

        return gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene

    def __len__(self):
        return len(self.poses_path_list)

    def parse_data_info(self):
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            start_frame = self.dataset_info['start_frame'][i]
            end_frame = self.dataset_info['end_frame'][i]
            scene = self.dataset_info['scene'][i]
            transform = self.dataset_info['transformation'][i]

            self.poses_path_list.append(start_frame)
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(scene)
            self.trans_path_list.append(transform)
            self.start_end_list.append([self.dataset_info['start_frame'][i], self.dataset_info['end_frame'][i]])

    def _get_raw_item(self, index):
        ego_idx = self.poses_path_list[index]
        scene = self.scenes_path_list[index]
        seq = self.sequences_path_list[index]
        start_frame, end_frame = self.start_end_list[index]

        poses_input_idx = []
        gazes = []
        gazes_mask = []
        poses_input = []

        transform_path = self.trans_path_list[index]
        transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
        scale = transform_info['scale']
        trans_pose2scene = np.array(transform_info['transformation'])
        trans_pose2scene[:3, 3] /= scale
        transform_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape(
            (4, 4))
        transform_norm[:3, 3] /= scale
        transform_pose = transform_norm @ trans_pose2scene

        for f in range(self.input_seq_len):
            pose_idx = ego_idx + int(f * 30 / self.fps)
            poses_input_idx.append(pose_idx)
            gaze_points = np.zeros((1, 3))
            gazes_mask.append(torch.zeros(1).long())

            gaze_ply_path = os.path.join(self.dataroot, scene, seq, 'eye_pc', '{}_center.ply'.format(pose_idx))
            gaze_pc_path = os.path.join(self.dataroot, scene, seq, 'eye_pc', '{}.ply'.format(pose_idx))

            if os.path.exists(gaze_pc_path):
                gaze_data = trimesh.load_mesh(gaze_ply_path)
                gaze_data.apply_scale(1 / scale)
                gaze_data.apply_transform(transform_norm)

                points = gaze_data.vertices
                if np.sum(abs(points)) > 1e-8:
                    gazes_mask[-1] = torch.ones(1).long()
                gaze_points = gaze_data.vertices[0:1]
                gaze_pc_data = trimesh.load_mesh(gaze_pc_path)
                if len(gaze_pc_data.vertices) == 0 or np.sum(abs(gaze_pc_data.vertices)) < 1e-8:
                    gazes_mask[-1] = torch.ones(0).long()

            pose_data = pickle.load(open(os.path.join(self.dataroot, scene, seq, 'smplx_local', '{}.pkl'.format(pose_idx)), 'rb'))
            ori = pose_data['orient'].detach().cpu().numpy()
            trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
            R = Rotation.from_rotvec(ori).as_matrix()

            R_s = transform_pose[:3, :3] @ R
            ori_s = Rotation.from_matrix(R_s).as_rotvec()
            trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

            poses_input.append(
                torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                           pose_data['latent']]))
            gazes.append(torch.from_numpy(gaze_points).float())

        gazes = torch.stack(gazes, dim=0)
        poses_input = torch.stack(poses_input, dim=0).detach()
        gazes_mask = torch.stack(gazes_mask, dim=0)

        gazes_valid_id = torch.where(gazes_mask)
        gazes_invalid_id = torch.where(torch.abs(gazes_mask - 1))
        gazes_valid = gazes[gazes_valid_id]
        gazes[gazes_invalid_id] *= 0
        gazes[gazes_invalid_id] += torch.mean(gazes_valid, dim=0, keepdim=True)

        poses_label = []
        poses_predict_idx = []
        for f in range(self.output_seq_len):
            pose_idx = ego_idx + int(self.input_seq_len * 30 / self.fps) + int(f * 30 / self.fps)
            poses_predict_idx.append(pose_idx)
            pose_path = os.path.join(self.dataroot, scene, seq, 'smplx_local',
                                     '{}.pkl'.format(pose_idx if f < self.output_seq_len else end_frame))
            pose_data = pickle.load(open(pose_path, 'rb'))

            ori = pose_data['orient'].detach().cpu().numpy()
            trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
            R = Rotation.from_rotvec(ori).as_matrix()
            R_s = transform_pose[:3, :3] @ R
            ori_s = Rotation.from_matrix(R_s).as_rotvec()
            trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

            poses_label.append(
                torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                           pose_data['latent']]))

        poses_label = torch.stack(poses_label, dim=0).detach()

        scene_points = self.scenes_raw[scene]
        transform_norm = torch.from_numpy(transform_norm).float()
        scene_points *= 1 / scale
        scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T
        scene_points = scene_points.cpu()

        return gazes, poses_input, poses_label, scene_points

    def _load_raw_scene(self, sample_points=32768):
        for i, scene_name in enumerate(self.dataset_info['scene']):
            if self.scenes_raw.get(scene_name, None) is None:
                print('loading scene of {}'.format(scene_name))
                scene_ply = trimesh.load_mesh(os.path.join(self.dataroot, scene_name, 'scene_obj', 'scene_downsampled.ply'))
                points = scene_ply.vertices[np.random.choice(range(len(scene_ply.vertices)), 65536)]
                with torch.no_grad():
                    points = torch.from_numpy(points).float().cuda().unsqueeze(0)
                    new_idx = pointnet2_utils.furthest_point_sample(points, sample_points).squeeze().long().cpu()
                    points = points.squeeze().cpu()[new_idx]
                self.scenes_raw[scene_name] = points.cpu()

    def _latent_to_joints(self, poses_full, vposer, body_model):
        joints_full = []

        for i in range(self.__len__()):
            poses_sample = poses_full[i]

            pose = {}
            body_pose = vposer.decode(poses_sample[:, 6:], output_type='aa')
            body_pose = body_pose.view(body_pose.shape[0], -1)  # bs * 21 * 3 -> bs * 63
            pose['body_pose'] = body_pose.cuda()
            pose['pose_embedding'] = poses_sample[:, 6:].cuda()
            pose['global_orient'] = poses_sample[:, :3].cuda()
            pose['transl'] = poses_sample[:, 3:6].cuda()
            smplx_output = body_model(return_verts=True, **pose)
            joints = smplx_output.joints
            joints_full.append(joints.detach().cpu())

        return torch.stack(joints_full, dim=0)

    def prepocess_dataset(self, folder, split):
        self.scenes_raw = dict()
        self._load_raw_scene(sample_points=self.config.sample_points)

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        vposer = vposer.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                  num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                  create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                  create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                  create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                  batch_size=self.config.output_seq_len + self.config.input_seq_len).cuda()

        gaze_full = []
        pose_input_full = []
        pose_label_full = []
        scene_points_full = []

        print('loading poses')
        for i in range(self.__len__()):
            gaze, pose_input, pose_label, scene_points = self._get_raw_item(i)
            gaze_full.append(gaze)
            pose_input_full.append(pose_input)
            pose_label_full.append(pose_label)
            scene_points_full.append(scene_points)

        gaze_full = torch.stack(gaze_full, dim=0)
        pose_input_full = torch.stack(pose_input_full, dim=0)
        pose_label_full = torch.stack(pose_label_full, dim=0)
        scene_points_full = torch.stack(scene_points_full, dim=0)

        pose_full = torch.cat([pose_input_full, pose_label_full], dim=1).cuda()
        joints = self._latent_to_joints(pose_full, vposer, body_model)
        joints_input, joints_label = joints[:, :self.config.input_seq_len], joints[:, self.config.input_seq_len:]

        data_dir = os.path.join(self.dataroot, folder, split)
        os.makedirs(data_dir, exist_ok=True)
        torch.save(gaze_full, os.path.join(data_dir, "gazes.pth"))
        torch.save(pose_input_full, os.path.join(data_dir, "poses_input.pth"))
        torch.save(pose_label_full, os.path.join(data_dir, "poses_label.pth"))
        torch.save(joints_input, os.path.join(data_dir, "joints_input.pth"))
        torch.save(joints_label, os.path.join(data_dir, "joints_label.pth"))
        torch.save(scene_points_full, os.path.join(data_dir, "scene_points_" + str(self.config.sample_points) + ".pth"))
