import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from utils.vis_utils import latent_to_joints
from model.base_cross_model import PerceiveEncoder
from model.pointnet_plus2 import PointNet2SemSegSSGShape, MyFPModule
from model.base_cross_model import SelfAttentionLayer


def human_centered_scene(xyz, ori, trans):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        ori: input global orient of the human, [B, len, 3]
        trans: input translation of the human, [B, len, 3]
    Return:
        new_xyz: normalized points position data, [B, len, N, 3]
    """
    bs, T, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    xyz = xyz[:, None, :, :].repeat(1, T, 1, 1) - trans[:, :, None, :]        # [B, len, N, 3]

    # First column is the X-axis transform, second for Y, and the third column for Z
    M = torch.cat([torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
                   torch.cat([torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)], dim=-1),
                   torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1)], dim=-1)
    M = M.view(bs, T, 3, 3)
    new_xyz = torch.matmul(xyz, M)

    return new_xyz


class SIF3D(nn.Module):
    def __init__(self, config, vposer, smplx_model):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.vposer = vposer
        self.smplx_model = smplx_model

        self.pointnet = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)         # joints (23*3) and smpl latents (3+3+32)
        self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim,
                                              n_latent=config.output_seq_len + config.input_seq_len,
                                              n_latent_channels=config.motion_latent_dim,
                                              n_self_att_heads=config.motion_n_heads,
                                              n_self_att_layers=config.motion_n_layers,
                                              dropout=config.dropout)

        self.gaze_fp_layer = MyFPModule()
        self.gaze_encoder = PerceiveEncoder(n_input_channels=config.scene_feats_dim,
                                            n_latent=config.output_seq_len + config.input_seq_len,
                                            n_latent_channels=config.gaze_hidden_dim,
                                            n_self_att_heads=config.motion_n_heads,
                                            n_self_att_layers=config.gaze_n_layers,
                                            dropout=config.dropout)

        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))

        self.tia_blocks = nn.ModuleList([TIABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)

        self.sca_blocks = nn.ModuleList([SCABlock(config.motion_latent_dim, num_head=4) for _ in range(1)])
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.joints_fc = nn.Linear(config.motion_latent_dim * 2, 69)
        self.motion_decoder = GCN(config, node_n=69)

    def forward(self, motions, joints, scene_xyz, gazes):
        """
        :param motions: (bs, seq_len, motion_dim)       [ori, trans, latent]
        :param scene_xyz: (bs, n, 3)
        :param gazes: (bs, seq_len, 1, 3)
        :return:
        """
        bs, seq_len, gaze_n, _ = gazes.shape
        assert gaze_n == 1
        gazes = gazes.squeeze(2)

        scene_feats, scene_global_feats = self.pointnet(scene_xyz.repeat(1, 1, 2))      # B x 256 x 30w, B x 256
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat([motions, joints.view(bs, seq_len, -1)], dim=-1)
        motions = self.motion_emb(motions)
        motions = self.motion_encoder(motions)
        gaze_embedding = self.gaze_fp_layer(gazes, scene_xyz, scene_feats.transpose(1, 2).contiguous()).transpose(1, 2)
        gaze_embedding = self.gaze_encoder(gaze_embedding)          # bs * len * c

        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_xyz, scene_feats, gaze_embedding)
        out_tia = out.clone()
        ori, trans = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)

        scene_xyz_normalized = human_centered_scene(scene_xyz, ori, trans)          # bs * len * N * 3
        dist_from_human = torch.norm(scene_xyz_normalized, dim=-1)
        dist_salience = -((dist_from_human * 1.5) ** 3)
        dir_salience = scene_xyz_normalized[:, :, :, 0] / torch.sqrt((scene_xyz_normalized[:, :, :, [0, 2]] ** 2).sum(dim=-1))
        spatial_prior = (dist_salience * self.dist_factor + dir_salience * self.dir_factor) + self.spatial_att(scene_xyz_normalized)

        out = motions.clone()
        for blk in self.sca_blocks:
            out = blk(out, scene_xyz, scene_feats, spatial_prior)
        out_sca = out.clone()
        pose = self.pose_fc(out_sca)

        latent = torch.cat([ori, trans, pose], dim=-1)
        recons_joints = latent_to_joints(latent, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()

        pred_joints_full = self.motion_decoder(recons_joints)[:, :, :23]

        return latent, pred_joints_full


class TIABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 3, feat_dim, drop=0.0)

    def forward(self, motion, scene_xyz, scene_feats, gaze_embedding):
        motion_tia = self.motion_encoder(motion)
        sm_feature = self.ia_attention(motion_tia, scene_feats)
        out = torch.cat([motion_tia, sm_feature, gaze_embedding], dim=-1)
        out = self.mlp(out)
        return out


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)        # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))       # bs * head * len * N
        att = F.softmax(att, dim=-1)                                # bs * head * N
        out = torch.matmul(att, v).view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_xyz, scene_feats, spatial_prior):
        motion_sca = self.motion_encoder(motion)
        sm_feature = self.sa_attention(motion_sca, scene_feats, spatial_prior)
        out = torch.cat([motion_sca, sm_feature], dim=-1)
        out = self.mlp(out)
        return out


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats, spatial_prior):
        bs, len, _ = motion.shape
        N = scene_feats.shape[1]

        q = self.motion_q(motion)           # bs * len * c
        q = q * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        # bs * head * len * _c, bs * head * N * _c
        q, k, v = q.view(bs, self.num_head, len, -1), k.view(bs, N, self.num_head, -1).transpose(1, 2), v.view(bs, N, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))       # bs * head * len * N
        att = att + spatial_prior[:, None, :, :]
        att = F.softmax(att, dim=-1)                         # bs * head * len * N
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(bs, len, -1)           # bs, len, c
        out = self.out_emb(out)

        return out


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz):
        out = F.relu(self.fc1(xyz))
        spatial_prior = self.fc2(out)
        return spatial_prior.squeeze(-1)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out
