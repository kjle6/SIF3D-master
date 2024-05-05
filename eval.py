import torch
import time
import smplx
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.sif3d import SIF3D
from utils.logger import MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.test_loader = DataLoader(
            gimo_dataset.EgoEvalDataset(config, train=False),
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.output_seq_len + config.input_seq_len,
                                       ).to(self.device)

    def eval(self):
        model = SIF3D(config, self.vposer, self.body_model).to(self.device)
        model = model.to(self.device)
        assert self.config.load_model_dir is not None
        print('loading pretrained model from ', self.config.load_model_dir)
        state_dict = torch.load(self.config.load_model_dir)
        state_dict = {k: v for k, v in state_dict.items() if "smplx" not in k}
        model.load_state_dict(state_dict, strict=False)
        print('load done!')

        with torch.no_grad():
            model.eval()
            test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

            for i, data in enumerate(self.test_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                poses_predict, joints_predict = model(poses_input, joints_input[:, :, :23], scene_points, gazes)

                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            print(test_metrics.result())
            test_metrics.reset()

    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label, test=False):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
        loss_des_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)[:, -1].mean()
        loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

        loss_all = F.l1_loss(poses_predict[:, :, :], poses_label[:, :, :], reduction='none')

        loss_ori = loss_all[:, :, :3]
        loss_trans = torch.norm(poses_predict[:, :, 3:6] - poses_label[:, :, 3:6], dim=-1)
        loss_latent = loss_all[:, :, 6:]

        pred_pose = self.vposer.decode(poses_predict[:, :, 6:], output_type='aa').view(-1, 63)
        pred_smplx = self.body_model(return_verts=True, body_pose=pred_pose,
                                     global_orient=poses_predict[:, :, :3].view(-1, 3),
                                     transl=poses_predict[:, :, 3:6].view(-1, 3),
                                     pose_embedding=poses_predict[:, :, 6:].view(-1, 32)).joints

        gt_smplx = torch.cat([joints_input, joints_label], dim=1)
        pred_smplx = pred_smplx.view(-1, self.config.input_seq_len + self.config.output_seq_len, 127, 3)[:, :, :23]
        gt_smplx -= gt_smplx[:, :, [0]]
        pred_smplx -= pred_smplx[:, :, [0]]

        mpjpe = torch.norm(gt_smplx - pred_smplx, dim=-1)

        return loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, mpjpe[:, -1].mean()

    def calc_loss_gcn(self, poses_predict, poses_label, poses_input):
        poses_label = torch.cat([poses_input, poses_label], dim=1)

        loss_trans = torch.norm(poses_predict[:, :, 0] - poses_label[:, :, 0], dim=-1)

        poses_label = poses_label - poses_label[:, :, [0]]
        poses_predict = poses_predict - poses_predict[:, :, [0]]
        mpjpe = torch.norm(poses_predict - poses_label, dim=-1)             # bs * seq * J

        return loss_trans, loss_trans[:, -1].mean(), mpjpe, mpjpe[:, -1].mean()


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    evaluator.eval()
