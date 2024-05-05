import os
import torch
import time
import smplx
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset import gimo_dataset
from human_body_prior.tools.model_loader import load_vposer
from config.config import MotionFromGazeConfig
from model.sif3d import SIF3D
from utils.logger import create_logger, MetricTracker


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = vposer.to(self.device)
        self.body_model = smplx.create(self.config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
                                       num_pca_comps=12, create_global_orient=True, create_body_pose=True,
                                       create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
                                       create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                       create_reye_pose=True, create_transl=True, num_betas=10, num_expression_coeffs=10,
                                       batch_size=config.batch_size * (config.output_seq_len + config.input_seq_len),
                                       ).to(self.device)

        self.model = SIF3D(config, self.vposer, self.body_model).to(self.device)

        if self.config.load_model_dir is not None:
            state_dict = torch.load(self.config.load_model_dir)
            state_dict = {k: v for k, v in state_dict.items() if ("smplx_model" not in k and "vposer" not in k)}
            self.model.load_state_dict(state_dict, strict=False)
            print('load done!')

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.lr_adjuster = ExponentialLR(self.optim, gamma=config.gamma)

        self.train_dataset = gimo_dataset.EgoEvalDataset(config, train=True)
        self.test_dataset = gimo_dataset.EgoEvalDataset(config, train=False)
        exit(0)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )
        # We set drop_last=True to fit the SMPL body model during training, test results could be a little inaccurate.
        # For accurate test results, run eval.sh with checkpoint.
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )

        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        self.logger = create_logger(config.save_path)
        os.makedirs(f"runs/{self.config.save_path}", exist_ok=True)

    def train(self):
        train_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for epoch in range(config.epoch):
            for data in tqdm(self.train_loader):
                gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene = data

                gazes = gazes.to(self.device)
                poses_input = poses_input.to(self.device)
                poses_label = poses_label.to(self.device)
                scene_points = scene_points.to(self.device).contiguous()
                joints_input = joints_input.to(self.device)
                joints_label = joints_label.to(self.device)

                poses_predict, joints_predict = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes)

                loss_ori, loss_trans, loss_latent, loss_des_ori, loss_des_trans, loss_des_latent, mpjpe, des_mpjpe = \
                    self.calculate_loss(poses_predict, poses_label, poses_input, joints_input[:, :, :23], joints_label[:, :, :23])
                loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                    self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

                loss = loss_ori.mean() + loss_trans.mean() + loss_latent.mean() + \
                       loss_des_ori + loss_des_trans + loss_des_latent + mpjpe.mean() + des_mpjpe + \
                       loss_trans_gcn.mean() + loss_des_trans_gcn + mpjpe_gcn.mean() + des_mpjpe_gcn

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
                train_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
                train_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

            self.lr_adjuster.step()
            train_metrics.log(self.logger, epoch)
            train_metrics.reset()

            if epoch % config.val_fre == 0:
                self.model.eval()
                with torch.no_grad():
                    self.test(epoch)
                self.model.train()
            if epoch % config.save_fre == 0:
                torch.save(self.model.state_dict(), f"{self.config.save_path}/{epoch}.pth")

    def test(self, epoch):
        test_metrics = MetricTracker('loss_trans', 'loss_des_trans', 'mpjpe', 'des_mpjpe')

        for i, data in enumerate(self.test_loader):
            gazes, poses_input, poses_label, joints_input, joints_label, scene_points, seq, scene = data

            gazes = gazes.to(self.device)
            poses_input = poses_input.to(self.device)
            poses_label = poses_label.to(self.device)
            scene_points = scene_points.to(self.device).contiguous()
            joints_input = joints_input.to(self.device)
            joints_label = joints_label.to(self.device)

            poses_predict, joints_predict = self.model(poses_input, joints_input[:, :, :23], scene_points, gazes)

            loss_trans_gcn, loss_des_trans_gcn, mpjpe_gcn, des_mpjpe_gcn = \
                self.calc_loss_gcn(joints_predict, joints_label[:, :, :23], joints_input[:, :, :23])

            test_metrics.update("loss_trans", loss_trans_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("loss_des_trans", loss_des_trans_gcn, gazes.shape[0])
            test_metrics.update("mpjpe", mpjpe_gcn[:, 6:].mean(), gazes.shape[0])
            test_metrics.update("des_mpjpe", des_mpjpe_gcn, gazes.shape[0])

        test_metrics.log(self.logger, epoch)
        test_metrics.reset()

    def calculate_loss(self, poses_predict, poses_label, poses_input, joints_input, joints_label):
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
    evaluator.train()
