import os
import torch
import numpy as np
from tqdm import tqdm

from utils.io_util import load_depth, load_rgb, glob_imgs
from utils.rend_util import rot_to_quat, load_K_Rt_from_P


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 downscale=1.,   # [H, W]
                 cam_file=None,
                 scale_radius=-1,
                 time_downsample_factor=30,
                 radius_init=1.,
                 start_moving=-1,
                 normalize_mode="shift_only"):

        assert os.path.exists(data_dir), "Data directory is empty"

        self.instance_dir = data_dir
        self.train_cameras = train_cameras

        # only camera projection matrix
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        if cam_file is not None:
            self.cam_file = '{0}/{1}'.format(self.instance_dir, cam_file)

        camera_dict = np.load(self.cam_file)
        # TODO: write different normalization methods to config file
        # camera projection matrix
        world_mats = camera_dict['world_mats'].astype(np.float32)
        # translation matrix to shift the ROI to coordinate centre
        normalize_mat = camera_dict['normalize_mat'].astype(np.float32)
        trans_mat = camera_dict['trans_mat']

        if normalize_mode == "shift_only":
            s = 1.
        else:
            s = normalize_mat[0, 0]

        self.intrinsics_all = []
        self.c2w_all = []
        cam_center_norms = []
        camera_ids = []
        for i, world_mat in enumerate(world_mats[:start_moving]):
            P = world_mat @ trans_mat if normalize_mode == "shift_only" else world_mat @ normalize_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(P)

            if np.linalg.norm(pose[:3, 3]) < radius_init:
                continue
            intrinsics_denorm, pose_denorm = load_K_Rt_from_P(world_mat[:3, :4])
            cam_center_norms.append(np.linalg.norm(pose[:3,3]))

            # downscale intrinsics
            intrinsics[0, 2] /= downscale
            intrinsics[1, 2] /= downscale
            intrinsics[0, 0] /= downscale
            intrinsics[1, 1] /= downscale
            # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.c2w_all.append(torch.from_numpy(pose).float())
            camera_ids += [i]
        self.intrinsics_all = self.intrinsics_all[::time_downsample_factor]
        self.c2w_all = self.c2w_all[::time_downsample_factor]
        max_cam_norm = max(cam_center_norms[::time_downsample_factor])

        image_dir = '{0}/rgb'.format(self.instance_dir)
        all_image_paths = sorted(glob_imgs(image_dir))
        image_paths = [all_image_paths[i] for i in camera_ids][::time_downsample_factor]
        depth_dir = '{0}/depth'.format(self.instance_dir)
        all_depth_paths = sorted(glob_imgs(depth_dir))
        depth_paths = [all_depth_paths[i] for i in camera_ids][::time_downsample_factor]

        self.n_images = len(image_paths)

        # determine width, height
        self.downscale = downscale
        tmp_rgb = load_rgb(image_paths[0], downscale)
        _, self.H, self.W = tmp_rgb.shape

        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm / 1.1)

        self.rgb_images = []
        for path in tqdm(image_paths, desc='loading images...'):
            rgb = load_rgb(path, downscale)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.depth_images = []
        for path in depth_paths:
            depth = load_depth(path, downscale)
            depth = depth.reshape(-1) / (5000. * s)
            depth[depth > 3.5] = -1.
            depth[depth < 0.05] = 0.
            self.depth_images.append(torch.from_numpy(depth).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)

        sample = dict()
        ground_truth = dict()

        ground_truth["rgb"] = self.rgb_images[idx]
        ground_truth["depth"] = self.depth_images[idx]
        sample["depth"] = self.depth_images[idx]
        sample["intrinsics"] = self.intrinsics_all[idx]

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_gt_pose(self):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        normalize_mat = camera_dict['normalize_mat'].astype(np.float32)
        c2w_all = []
        for world_mat in world_mats:
            P = world_mat @ normalize_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            c2w_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = load_K_Rt_from_P(P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat


if __name__ == "__main__":
    dataset = SceneDataset(False, '../data/tum/fr3_long_office/processed')
    c2w = dataset.get_gt_pose().data.cpu().numpy()
    from tools.vis_camera import visualize
    visualize(c2w)