from superglue_models.superpoint import SuperPoint
import argparse
import torch
import tqdm
import os
import cv2
import numpy as np
import open3d as o3d
import random
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, distance_transform_edt
import matplotlib.pyplot as plt
from gomatch.utils.extract_matches import align_points2d


def show_keypoints(image, kpts, only_kpts=False):
    # image is assumed to have shape (H, W, 3) and kpts is shape (N, 2)
    vis_image = image.mean(-1).cpu().numpy()
    H, W = vis_image.shape
    vis_image = (vis_image * 255).astype(np.uint8)

    if only_kpts:
        out = np.zeros_like(vis_image)
    else:
        out = np.copy(vis_image)
    kpts_x = kpts[:, 0].long().cpu().numpy()
    kpts_y = kpts[:, 1].long().cpu().numpy()

    white = (255, 255, 255)
    black = (0, 0, 0)

    for x, y in zip(kpts_x, kpts_y):
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    
    
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.show()


def show_matches(image0, image1, kpts_sphere0, kpts_sphere1, kpts_ij0, kpts_ij1, show_keypoints=True, thres=0.1, margin=10):
    # image is assumed to have shape (H, W, 3) and kpts is shape (N, 2)
    vis_image0 = image0.mean(-1).cpu().numpy()
    H0, W0 = vis_image0.shape
    vis_image0 = (vis_image0 * 255).astype(np.uint8)

    vis_image1 = image1.mean(-1).cpu().numpy()
    H1, W1 = vis_image1.shape
    vis_image1 = (vis_image1 * 255).astype(np.uint8)

    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = vis_image0
    out[:H1, W0+margin:] = vis_image1
    out = np.stack([out] * 3, -1)
    
    kpts_x0 = kpts_ij0[:, 0].long().cpu().numpy()
    kpts_y0 = kpts_ij0[:, 1].long().cpu().numpy()
    kpts_x1 = kpts_ij1[:, 0].long().cpu().numpy()
    kpts_y1 = kpts_ij1[:, 1].long().cpu().numpy()

    if show_keypoints:
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in zip(kpts_x0, kpts_y0):
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in zip(kpts_x1, kpts_y1):
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    sphere_match = align_points2d(kpts_sphere0.cpu().numpy(), kpts_sphere1.cpu().numpy(), thres)
    i2ds, i3ds = sphere_match[:, 0], sphere_match[:, 1]

    mkpts_x0 = kpts_x0[i2ds]
    mkpts_y0 = kpts_y0[i2ds]
    mkpts_x1 = kpts_x1[i3ds]
    mkpts_y1 = kpts_y1[i3ds]

    c = (0, 255, 0)
    for idx, (x0, y0, x1, y1) in enumerate(zip(mkpts_x0, mkpts_y0, mkpts_x1, mkpts_y1)):
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
            color=c, thickness=1, lineType=cv2.LINE_AA)

        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
            lineType=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.show()


def rgb_to_grayscale(
    image: torch.Tensor, rgb_weights: torch.Tensor = torch.tensor([0.299, 0.587, 0.114])
) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if not isinstance(rgb_weights, torch.Tensor):
        raise TypeError(f"rgb_weights is not a torch.Tensor. Got {type(rgb_weights)}")

    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"rgb_weights must have a shape of (*, 3). Got {rgb_weights.shape}")

    r: torch.Tensor = image[..., 0:1, :, :]
    g: torch.Tensor = image[..., 1:2, :, :]
    b: torch.Tensor = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise TypeError(
            f"Input image and rgb_weights should be of same dtype. Got {image.dtype} and {rgb_weights.dtype}"
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b


def sample_from_img(img, coord_arr, padding='zeros', mode='bilinear', batched=False) -> torch.Tensor:
    """
    Image sampling function
    Use coord_arr as a grid for sampling from img

    Args:
        img: (H, W, 3) torch tensor containing image RGB values
        coord_arr: (N, 2) torch tensor containing image coordinates, ranged in [-1, 1], converted from 3d coordinates
        padding: Padding mode to use for grid_sample
        mode: How to sample from grid
        batched: If True, assumes an additional batch dimension for coord_arr

    Returns:
        sample_rgb: (N, 3) torch tensor containing sampled RGB values
    """
    if batched:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(coord_arr.shape[0], coord_arr.shape[1], 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img.expand(coord_arr.shape[0], -1, -1, -1), sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(sample_rgb)  # (B, 3, N)
        sample_rgb = torch.transpose(sample_rgb, 1, 2)  # (B, N, 3)  

    else:
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)

        # sampling from img
        sample_arr = coord_arr.reshape(1, -1, 1, 2)
        sample_arr = torch.clip(sample_arr, min=-0.99, max=0.99)
        sample_rgb = F.grid_sample(img, sample_arr, mode=mode, align_corners=False, padding_mode=padding)

        sample_rgb = torch.squeeze(torch.squeeze(sample_rgb, 0), 2)
        sample_rgb = torch.transpose(sample_rgb, 0, 1)

    return sample_rgb


def inv_cloud2idx(coord_arr):
    # Inversion of cloud2idx: given a (N, 2) coord_arr, returns a set of (N, 3) 3D points on a sphere.
    sphere_cloud_arr = (coord_arr + 1.) / 2.
    sphere_cloud_arr[:, 0] = (1.0 - sphere_cloud_arr[:, 0]) * (2 * np.pi)
    sphere_cloud_arr[:, 1] = np.pi * sphere_cloud_arr[:, 1]  # Contains [phi, theta] of sphere

    sphere_cloud_arr[:, 0] -= np.pi  # Subtraction to accomodate for cloud2idx

    sphere_xyz = torch.zeros(sphere_cloud_arr.shape[0], 3, device=coord_arr.device)
    sphere_xyz[:, 0] = torch.sin(sphere_cloud_arr[:, 1]) * torch.cos(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 1] = torch.sin(sphere_cloud_arr[:, 1]) * torch.sin(sphere_cloud_arr[:, 0])
    sphere_xyz[:, 2] = torch.cos(sphere_cloud_arr[:, 1])

    return sphere_xyz


def cloud2idx(xyz, batched = False) -> torch.Tensor:
    """
    Change 3d coordinates to image coordinates ranged in [-1, 1].

    Args:
        xyz: (N, 3) torch tensor containing xyz values of the point cloud data
        batched: If True, performs batched operation with xyz considered as shape (B, N, 3)

    Returns:
        coord_arr: (N, 2) torch tensor containing transformed image coordinates
    """
    if batched:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[..., :2], dim=-1)), xyz[..., 2] + 1e-6), -1)  # (B, N, 1)

        # horizontal angle
        phi = torch.atan2(xyz[..., 1:2], xyz[..., 0:1] + 1e-6)  # (B, N, 1)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)  # (B, N, 2)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[..., 0] / (np.pi * 2), sphere_cloud_arr[..., 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)  # (B, N, 2)

    else:
        # first project 3d coordinates to a unit sphere and obtain vertical/horizontal angle

        # vertical angle
        theta = torch.unsqueeze(torch.atan2((torch.norm(xyz[:, :2], dim=-1)), xyz[:, 2] + 1e-6), 1)

        # horizontal angle
        phi = torch.atan2(xyz[:, 1:2], xyz[:, 0:1] + 1e-6)
        phi += np.pi

        sphere_cloud_arr = torch.cat([phi, theta], dim=-1)

        # image coordinates ranged in [0, 1]
        coord_arr = torch.stack([1.0 - sphere_cloud_arr[:, 0] / (np.pi * 2), sphere_cloud_arr[:, 1] / np.pi], dim=-1)
        # Rearrange so that the range is in [-1, 1]
        coord_arr = (2 * coord_arr - 1)

    return coord_arr


def ij2coord(ij_values, resolution):
    # Convert (N, 2) image ij-coordinates to 3D spherical coordinates
    coord_idx = torch.flip(ij_values.float(), [-1])
    coord_idx[:, 0] /= (resolution[1] - 1)
    coord_idx[:, 1] /= (resolution[0] - 1)

    coord_idx = 2. * coord_idx - 1.

    sphere_xyz = inv_cloud2idx(coord_idx)  # Points on sphere
    return sphere_xyz


def make_pano(xyz, rgb, resolution = (200, 400), 
        return_torch = False, return_coord = False, return_norm_coord = False, 
        default_white=False, fill_hole=False):
    """
    Make panorama image from xyz and rgb tensors

    Args:
        xyz: (N, 3) torch tensor containing xyz coordinates
        rgb: (N, 3) torch tensor containing rgb values, ranged in [0, 1]
        resolution: Tuple size of 2, returning panorama image of size resolution
        return_torch: if True, return image as torch.Tensor
                      if False, return image as numpy.array
        return_coord: If True, return coordinate in long format
        return_norm_coord: If True, return coordinate in normalized float format
        default_white: If True, defaults the color values to white
        fill_hole: If True, fills holes from rendering

    Returns:
        image: (H, W, 3) torch.Tensor or numpy.array
    """

    with torch.no_grad():

        # project farther points first
        dist = torch.norm(xyz, dim=-1)
        mod_idx = torch.argsort(dist)
        mod_idx = torch.flip(mod_idx, dims=[0])
        mod_xyz = xyz.clone().detach()[mod_idx]
        mod_rgb = rgb.clone().detach()[mod_idx]

        orig_coord_idx = cloud2idx(mod_xyz)
        coord_idx = (orig_coord_idx + 1.0) / 2.0
        # coord_idx[:, 0] is x coordinate, coord_idx[:, 1] is y coordinate
        coord_idx[:, 0] *= (resolution[1] - 1)
        coord_idx[:, 1] *= (resolution[0] - 1)

        coord_idx = torch.flip(coord_idx, [-1])
        coord_idx = coord_idx.long()
        save_coord_idx = coord_idx.clone().detach()
        coord_idx = tuple(coord_idx.t())

        if default_white:
            image = torch.ones([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)
        else:
            image = torch.zeros([resolution[0], resolution[1], 3], dtype=torch.float, device=xyz.device)

        # color the image
        # pad by 1
        temp = torch.ones_like(coord_idx[0], device=xyz.device)
        coord_idx1 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx2 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      coord_idx[1])
        coord_idx3 = (torch.clamp(coord_idx[0] + temp, max=resolution[0] - 1),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx4 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx5 = (torch.clamp(coord_idx[0] - temp, min=0),
                      coord_idx[1])
        coord_idx6 = (torch.clamp(coord_idx[0] - temp, min=0),
                      torch.clamp(coord_idx[1] - temp, min=0))
        coord_idx7 = (coord_idx[0],
                      torch.clamp(coord_idx[1] + temp, max=resolution[1] - 1))
        coord_idx8 = (coord_idx[0],
                      torch.clamp(coord_idx[1] - temp, min=0))

        image.index_put_(coord_idx8, mod_rgb, accumulate=False)
        image.index_put_(coord_idx7, mod_rgb, accumulate=False)
        image.index_put_(coord_idx6, mod_rgb, accumulate=False)
        image.index_put_(coord_idx5, mod_rgb, accumulate=False)
        image.index_put_(coord_idx4, mod_rgb, accumulate=False)
        image.index_put_(coord_idx3, mod_rgb, accumulate=False)
        image.index_put_(coord_idx2, mod_rgb, accumulate=False)
        image.index_put_(coord_idx1, mod_rgb, accumulate=False)
        image.index_put_(coord_idx, mod_rgb, accumulate=False)

        image = image * 255
        
        if fill_hole:
            image_copy = image.cpu().numpy()
            fill_idx = distance_transform_edt((image_copy.sum(-1) == 0), return_distances=False, return_indices=True)
            image_copy = image_copy[tuple(fill_idx)]
            image = torch.from_numpy(image_copy).to(xyz.device)

        if not return_torch:
            image = image.cpu().numpy().astype(np.uint8)
    if return_coord:
        # mod_idx is in (i, j) format, not (x, y) format
        inv_mod_idx = torch.argsort(mod_idx)
        return image, save_coord_idx[inv_mod_idx]
    elif return_norm_coord:
        inv_mod_idx = torch.argsort(mod_idx)
        return image, orig_coord_idx[inv_mod_idx]
    else:
        return image


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw[..., None]
        pitch = pitch[..., None]
        roll = roll[..., None]

        tensor_0 = np.zeros(1)
        tensor_1 = np.ones(1)

        RX = np.stack([
                        np.stack([tensor_1, tensor_0, tensor_0]),
                        np.stack([tensor_0, np.cos(roll), -np.sin(roll)]),
                        np.stack([tensor_0, np.sin(roll), np.cos(roll)])]).reshape(3, 3)

        RY = np.stack([
                        np.stack([np.cos(pitch), tensor_0, np.sin(pitch)]),
                        np.stack([tensor_0, tensor_1, tensor_0]),
                        np.stack([-np.sin(pitch), tensor_0, np.cos(pitch)])]).reshape(3, 3)

        RZ = np.stack([
                        np.stack([np.cos(yaw), -np.sin(yaw), tensor_0]),
                        np.stack([np.sin(yaw), np.cos(yaw), tensor_0]),
                        np.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = RZ @ RY
        R = R @ RX

        return R
    
    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return np.stack(tot_mtx)


def ypr_from_rot(rot_mtx):
    def _mtx2ypr(in_mtx):
        # in_mtx is assumed to have a shape of [3, 3]
        yaw = np.arctan2(in_mtx[1, 0], in_mtx[0, 0] + 1e-6)
        pitch = np.arcsin(-in_mtx[2, 0])
        roll = np.arctan2(in_mtx[2, 1], in_mtx[2, 2])

        ypr = np.array([yaw, pitch, roll])
        return ypr
    
    if len(rot_mtx.shape) == 2:
        return _mtx2ypr(rot_mtx)
    else:
        tot_mtx = []
        for mtx in rot_mtx:
            tot_mtx.append(_mtx2ypr(mtx))
        return np.stack(tot_mtx)


def read_gt_pose(filename, dataset):
    # Return (3, 1) gt_trans and (3, 3) gt_rot as numpy array
    if dataset == 'mp3d':
        pose = np.loadtxt(filename)

        # ground truth translation
        gt_trans = pose[:, 3][:3].reshape((3, 1))

        # ground truth rotation
        rot = pose[:3, :3]
        rot = rot.T
        rot[:, :] = rot[[2, 0, 1], :]
        theta = np.pi / 6
        rot_mat = np.array([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
        rot_mat_2 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rot = np.matmul(rot_mat, rot)
        gt_rot = np.matmul(rot_mat_2, rot)

    else:
        raise NotImplementedError("Other datasets are not supported")

    return gt_trans, gt_rot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Type of dataset to use", default="mp3d")
    parser.add_argument("--data_root", help="Root directory containing dataset", default="./data/matterport")
    parser.add_argument("--save_root", help="Root directory containing generated keypoints", default="./data/matterport_kpts")
    parser.add_argument("--render_size", help="Size of rendered panorama provided as (height, width)", default=[320, 640])
    parser.add_argument("--trans_bound", help="Size of translation bound for generating pairs as (size_x, size_y, size_z)", default=[0.5, 0.5, 0.25])
    parser.add_argument("--rot_bound", help="Size of rotation bound for generating pairs as (size_yaw, size_pitch, size_roll)", default=[30, 15, 10])
    parser.add_argument("--num_trans", help="Number of translation poses to spawn per x and y", default=10)
    parser.add_argument("--num_pairs", help="Number of reference pose pairs to generate per translation location", default=1)
    parser.add_argument("--vis_match", help="Visualize matches for debugging", action='store_true')
    args = parser.parse_args()

    detector_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    }
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    detector = SuperPoint(detector_config).to(device)

    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)
    else:
        raise FileExistsError("Directory already exists")
    
    global_id = -1
    min_kpts = 10

    if args.dataset == 'mp3d':
        scene_list = [os.path.join(args.data_root, d) for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
        for scene_idx, scene in enumerate(scene_list):
            if not os.path.exists(os.path.join(args.save_root, scene.split('/')[-1])):
                os.mkdir(os.path.join(args.save_root, scene.split('/')[-1]))
            else:
                raise FileExistsError("Directory already exists")

            print(f"Scene Name (No. {scene_idx}): {scene.split('/')[-1]}")
            space_list = sorted([s.strip('.pcd') for s in os.listdir(os.path.join(scene, 'pcd'))])
            for space in space_list:
                print(f"Processing room {scene.split('/')[-1]}/{space}...")
                if not os.path.exists(os.path.join(args.save_root, scene.split('/')[-1], space)):
                    os.mkdir(os.path.join(args.save_root, scene.split('/')[-1], space))
                else:
                    raise FileExistsError("Directory already exists")

                pcd_name = os.path.join(scene, 'pcd', space + '.pcd')
                pcd = o3d.io.read_point_cloud(pcd_name)
                xyz_np = np.asarray(pcd.points, dtype=np.float32)
                rgb_np = np.asarray(pcd.colors, dtype=np.float32)
                xyz = torch.from_numpy(xyz_np).to(device)
                rgb = torch.from_numpy(rgb_np).to(device)

                pose_file_list = [os.path.join(scene, 'pose', space, p) for p in os.listdir(os.path.join(scene, 'pose', space))]
                pano_file_list = [p.replace('txt', 'jpg').replace('/pose', '/pano') for p in pose_file_list]
                
                pose_file_list = sorted(pose_file_list)
                pano_file_list = sorted(pano_file_list)

                # Collect pose annotations and generate bounding boxes from them
                annotated_trans = []
                annotated_rot = []
                for pose_file in pose_file_list:
                    gt_trans, gt_rot = read_gt_pose(pose_file, args.dataset)
                    annotated_trans.append(gt_trans)
                    annotated_rot.append(gt_rot)
                annotated_trans = np.concatenate(annotated_trans, axis=-1).T  # (N_annot, 3)
                annotated_rot = np.stack(annotated_rot, axis=0)  # (N_annot, 3, 3)
                

                # Create (real, synthetic) pairs
                print("Creating (real, synthetic) pairs...")
                for pano_file, trans, rot in tqdm.tqdm(zip(pano_file_list, annotated_trans, annotated_rot), total=len(pano_file_list)):
                    for repeat in range(args.num_pairs):
                        global_id += 1

                        # Make panorama at (trans, rot)
                        trans_tensor = torch.from_numpy(trans).unsqueeze(0).to(device)
                        rot_tensor = torch.from_numpy(rot).to(device)

                        pano = cv2.cvtColor(cv2.imread(pano_file), cv2.COLOR_BGR2RGB)
                        pano = cv2.resize(pano, (args.render_size[1], args.render_size[0]))
                        pano = torch.from_numpy(pano).float().to(device) / 255.
                        
                        rand_trans = np.array(args.trans_bound) * (2 * np.random.random([trans.shape[0]]) - 1)
                        new_trans = trans + rand_trans

                        rand_yaw = np.deg2rad(args.rot_bound[0]) * random.random()
                        rand_pitch = np.deg2rad(args.rot_bound[1]) * (2 * random.random() - 1)
                        rand_roll = np.deg2rad(args.rot_bound[2]) * (2 * random.random() - 1)

                        rand_ypr = np.array([rand_yaw, rand_pitch, rand_roll])
                        rand_rot = rot_from_ypr(rand_ypr)
                        new_rot = rot @ rand_rot

                        # Make panorama at (new_trans, new_rot)
                        new_trans_tensor = torch.from_numpy(new_trans).unsqueeze(0).to(device)
                        new_rot_tensor = torch.from_numpy(new_rot).to(device)
                        new_pano = make_pano((xyz - new_trans_tensor) @ new_rot_tensor.T, rgb, args.render_size, return_torch=True).float() / 255.  # (H, W, 3)

                        # Extract keypoints
                        pano_input = rgb_to_grayscale(pano.permute(2, 0, 1).unsqueeze(0))
                        new_pano_input = rgb_to_grayscale(new_pano.permute(2, 0, 1).unsqueeze(0))

                        with torch.no_grad():
                            pano_pred = detector({'image': pano_input})
                            new_pano_pred = detector({'image': new_pano_input})

                            pano_kpts_ij = pano_pred['keypoints'][0]
                            new_pano_kpts_ij = new_pano_pred['keypoints'][0]

                            # Check validity of keypoints
                            if pano_kpts_ij.shape[0] < min_kpts or new_pano_kpts_ij.shape[0] < min_kpts:
                                global_id -= 1
                                continue

                            pano_kpts_sphere = ij2coord(torch.flip(pano_pred['keypoints'][0], [-1]), args.render_size)
                            new_pano_kpts_sphere = ij2coord(torch.flip(new_pano_pred['keypoints'][0], [-1]), args.render_size)

                            coord_pano = make_pano((xyz - trans_tensor) @ rot_tensor.T, xyz, return_torch=True, resolution=args.render_size).float() / 255.
                            new_coord_pano = make_pano((xyz - new_trans_tensor) @ new_rot_tensor.T, xyz, return_torch=True, resolution=args.render_size).float() / 255.
                            
                            # Extract 3D points
                            pano_kpts_ij_sample = pano_kpts_ij.clone()
                            pano_kpts_ij_sample[:, 0] = (pano_kpts_ij_sample[:, 0] - args.render_size[1] // 2) / (args.render_size[1] // 2)
                            pano_kpts_ij_sample[:, 1] = (pano_kpts_ij_sample[:, 1] - args.render_size[0] // 2) / (args.render_size[0] // 2)
                            kpts_xyz = sample_from_img(coord_pano, pano_kpts_ij_sample, mode='nearest')
                            valid_mask = kpts_xyz.norm(dim=-1) > 0.1
                            pano_kpts_3d = kpts_xyz[valid_mask]
                            pano_kpts_sphere = pano_kpts_sphere[valid_mask]
                            pano_kpts_ij = pano_kpts_ij[valid_mask]

                            new_pano_kpts_ij_sample = new_pano_kpts_ij.clone()
                            new_pano_kpts_ij_sample[:, 0] = (new_pano_kpts_ij_sample[:, 0] - args.render_size[1] // 2) / (args.render_size[1] // 2)
                            new_pano_kpts_ij_sample[:, 1] = (new_pano_kpts_ij_sample[:, 1] - args.render_size[0] // 2) / (args.render_size[0] // 2)
                            new_kpts_xyz = sample_from_img(new_coord_pano, new_pano_kpts_ij_sample, mode='nearest')
                            new_valid_mask = new_kpts_xyz.norm(dim=-1) > 0.1
                            new_pano_kpts_3d = new_kpts_xyz[new_valid_mask]
                            new_pano_kpts_sphere = new_pano_kpts_sphere[new_valid_mask]
                            new_pano_kpts_ij = new_pano_kpts_ij[new_valid_mask]

                            # Optionally visualize
                            if args.vis_match:
                                new_pano_kpts_3d_trans = (new_pano_kpts_3d - trans_tensor) @ rot_tensor.T
                                new_pano_kpts_3d_proj = new_pano_kpts_3d_trans / new_pano_kpts_3d_trans.norm(dim=-1, keepdim=True)
                                show_matches(pano, new_pano, pano_kpts_sphere, new_pano_kpts_3d_proj, pano_kpts_ij, new_pano_kpts_ij)

                            # Check validity of far range keypoints
                            if valid_mask.sum() < min_kpts or new_valid_mask.sum() < min_kpts:
                                global_id -= 1
                                continue
                        
                        data_orig = {'R': rot, 'T': trans, 'kpts_sphere': pano_kpts_sphere.cpu().numpy(),
                                    'kpts_ij': pano_kpts_ij.cpu().numpy(), 'kpts_3d': pano_kpts_3d.cpu().numpy()}
                        data_new = {'R': new_rot, 'T': new_trans, 'kpts_sphere': new_pano_kpts_sphere.cpu().numpy(),
                                    'kpts_ij': new_pano_kpts_ij.cpu().numpy(), 'kpts_3d': new_pano_kpts_3d.cpu().numpy()}
                        
                        data = {**data_orig, **{k+'_pair': data_new[k] for k in data_new.keys()}}
                        
                        data_dir = os.path.join(args.save_root, scene.split('/')[-1], space, f"{global_id}.npz")
                        np.savez(data_dir, **data)


                # Create (synthetic, synthetic) pairs
                print("Creating (synthetic, synthetic) pairs...")
                min_x, min_y, min_z = annotated_trans.min(0)
                max_x, max_y, max_z = annotated_trans.max(0)

                bound_min_x = min_x - args.trans_bound[0]
                bound_max_x = max_x + args.trans_bound[0]
                bound_min_y = min_y - args.trans_bound[1]
                bound_max_y = max_y + args.trans_bound[1]
                bound_min_z = min_z - args.trans_bound[2]
                bound_max_z = max_z + args.trans_bound[2]

                unif_x = np.linspace(bound_min_x, bound_max_x, args.num_trans)
                unif_y = np.linspace(bound_min_y, bound_max_y, args.num_trans)
                unif_z = np.linspace(bound_min_z, bound_max_z, 3)

                trans_coords = np.meshgrid(unif_x, unif_y, unif_z)
                trans_arr = np.stack([trans_coords[0].reshape(-1), trans_coords[1].reshape(-1), trans_coords[2].reshape(-1)]).T  # (N_trans, 3)

                for trans in tqdm.tqdm(trans_arr):
                    for repeat in range(args.num_pairs):
                        global_id += 1
                        # Create random rotation view for each translation
                        yaw = 2 * np.pi * random.random()
                        pitch = np.deg2rad(args.rot_bound[1]) * (2 * random.random() - 1)
                        roll = np.deg2rad(args.rot_bound[2]) * (2 * random.random() - 1)

                        ypr = np.array([yaw, pitch, roll])
                        rot = rot_from_ypr(ypr)

                        # Make panorama at (trans, rot)
                        trans_tensor = torch.from_numpy(trans).unsqueeze(0).to(device)
                        rot_tensor = torch.from_numpy(rot).to(device)
                        pano = make_pano((xyz - trans_tensor) @ rot_tensor.T, rgb, args.render_size, return_torch=True).float() / 255.  # (H, W, 3)

                        rand_trans = np.array(args.trans_bound) * (2 * np.random.random([trans.shape[0]]) - 1)
                        new_trans = trans + rand_trans

                        rand_yaw = np.deg2rad(args.rot_bound[0]) * random.random()
                        rand_pitch = np.deg2rad(args.rot_bound[1]) * (2 * random.random() - 1)
                        rand_roll = np.deg2rad(args.rot_bound[2]) * (2 * random.random() - 1)

                        rand_ypr = np.array([rand_yaw, rand_pitch, rand_roll])
                        rand_rot = rot_from_ypr(rand_ypr)
                        new_rot = rot @ rand_rot

                        # Make panorama at (new_trans, new_rot)
                        new_trans_tensor = torch.from_numpy(new_trans).unsqueeze(0).to(device)
                        new_rot_tensor = torch.from_numpy(new_rot).to(device)
                        new_pano = make_pano((xyz - new_trans_tensor) @ new_rot_tensor.T, rgb, args.render_size, return_torch=True).float() / 255.  # (H, W, 3)

                        # Extract keypoints
                        pano_input = rgb_to_grayscale(pano.permute(2, 0, 1).unsqueeze(0))
                        new_pano_input = rgb_to_grayscale(new_pano.permute(2, 0, 1).unsqueeze(0))

                        with torch.no_grad():
                            pano_pred = detector({'image': pano_input})
                            new_pano_pred = detector({'image': new_pano_input})

                            pano_kpts_ij = pano_pred['keypoints'][0]
                            new_pano_kpts_ij = new_pano_pred['keypoints'][0]

                            # Check validity of keypoints
                            if pano_kpts_ij.shape[0] < min_kpts or new_pano_kpts_ij.shape[0] < min_kpts:
                                global_id -= 1
                                continue

                            pano_kpts_sphere = ij2coord(torch.flip(pano_pred['keypoints'][0], [-1]), args.render_size)
                            new_pano_kpts_sphere = ij2coord(torch.flip(new_pano_pred['keypoints'][0], [-1]), args.render_size)

                            coord_pano = make_pano((xyz - trans_tensor) @ rot_tensor.T, xyz, return_torch=True, resolution=args.render_size).float() / 255.
                            new_coord_pano = make_pano((xyz - new_trans_tensor) @ new_rot_tensor.T, xyz, return_torch=True, resolution=args.render_size).float() / 255.
                            
                            # Extract 3D points
                            pano_kpts_ij_sample = pano_kpts_ij.clone()
                            pano_kpts_ij_sample[:, 0] = (pano_kpts_ij_sample[:, 0] - args.render_size[1] // 2) / (args.render_size[1] // 2)
                            pano_kpts_ij_sample[:, 1] = (pano_kpts_ij_sample[:, 1] - args.render_size[0] // 2) / (args.render_size[0] // 2)
                            kpts_xyz = sample_from_img(coord_pano, pano_kpts_ij_sample, mode='nearest')
                            valid_mask = kpts_xyz.norm(dim=-1) > 0.1
                            pano_kpts_3d = kpts_xyz[valid_mask]
                            pano_kpts_sphere = pano_kpts_sphere[valid_mask]
                            pano_kpts_ij = pano_kpts_ij[valid_mask]

                            new_pano_kpts_ij_sample = new_pano_kpts_ij.clone()
                            new_pano_kpts_ij_sample[:, 0] = (new_pano_kpts_ij_sample[:, 0] - args.render_size[1] // 2) / (args.render_size[1] // 2)
                            new_pano_kpts_ij_sample[:, 1] = (new_pano_kpts_ij_sample[:, 1] - args.render_size[0] // 2) / (args.render_size[0] // 2)
                            new_kpts_xyz = sample_from_img(new_coord_pano, new_pano_kpts_ij_sample, mode='nearest')
                            new_valid_mask = new_kpts_xyz.norm(dim=-1) > 0.1
                            new_pano_kpts_3d = new_kpts_xyz[new_valid_mask]
                            new_pano_kpts_sphere = new_pano_kpts_sphere[new_valid_mask]
                            new_pano_kpts_ij = new_pano_kpts_ij[new_valid_mask]

                            # Optionally visualize
                            if args.vis_match:
                                new_pano_kpts_3d_trans = (new_pano_kpts_3d - trans_tensor) @ rot_tensor.T
                                new_pano_kpts_3d_proj = new_pano_kpts_3d_trans / new_pano_kpts_3d_trans.norm(dim=-1, keepdim=True)
                                show_matches(pano, new_pano, pano_kpts_sphere, new_pano_kpts_3d_proj, pano_kpts_ij, new_pano_kpts_ij)

                            # Check validity of far range keypoints
                            if valid_mask.sum() < min_kpts or new_valid_mask.sum() < min_kpts:
                                global_id -= 1
                                continue

                        data_orig = {'R': rot, 'T': trans, 'kpts_sphere': pano_kpts_sphere.cpu().numpy(),
                                    'kpts_ij': pano_kpts_ij.cpu().numpy(), 'kpts_3d': pano_kpts_3d.cpu().numpy()}
                        data_new = {'R': new_rot, 'T': new_trans, 'kpts_sphere': new_pano_kpts_sphere.cpu().numpy(),
                                    'kpts_ij': new_pano_kpts_ij.cpu().numpy(), 'kpts_3d': new_pano_kpts_3d.cpu().numpy()}
                        
                        data = {**data_orig, **{k+'_pair': data_new[k] for k in data_new.keys()}}
                        
                        data_dir = os.path.join(args.save_root, scene.split('/')[-1], space, f"{global_id}.npz")
                        np.savez(data_dir, **data)

    else:
        raise NotImplementedError("Other datasets are not supported yet")
