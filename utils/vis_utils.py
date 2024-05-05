import torch
import torch.nn as nn


def latent_to_joints(pose_latent, vposer, smplx_model):
    bs, T, c = pose_latent.shape
    body_pose = vposer.decode(pose_latent[:, :, 6:], output_type='aa').view(-1, 63)
    smplx_output = smplx_model(return_verts=True, body_pose=body_pose,
                               global_orient=pose_latent[:, :, :3].view(-1, 3),
                               transl=pose_latent[:, :, 3:6].view(-1, 3),
                               pose_embedding=pose_latent[:, :, 6:].view(-1, 32))
    joints = smplx_output.joints
    return joints.reshape(bs, T, 127, 3)
