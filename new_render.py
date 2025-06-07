#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
from utils.render_utils import error_map_colors, validate_override_color

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def overlay_heatmap(gray: np.ndarray,
                    heatmap: np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    """
    Overlay a BGR “hot” heatmap over a single‐channel grayscale image,
    producing the same style of relevancy‐map overlay as LERF.

    Args:
        gray (np.ndarray):         Grayscale image of shape (H, W) or (H, W, 1), dtype uint8 or float32/float64.
        heatmap (np.ndarray):      “Hot”‐coded color map of shape (H, W, 3), dtype uint8 or float32/float64, in BGR.
        eps (float, optional):     Tiny constant to avoid division by zero when normalizing. Defaults to 1e-6.

    Returns:
        overlay (np.ndarray):      BGR image of shape (H, W, 3), same dtype as input, with heatmap overlaid on top of grayscale.
    """

    # 1. Ensure gray is shape (H, W); if (H, W, 1), squeeze it
    if gray.ndim == 3 and gray.shape[2] == 1:
        gray = gray[:, :, 0]
    elif gray.ndim != 2:
        raise ValueError(f"Expected gray of shape (H, W) or (H, W, 1), got {gray.shape}.")

    # 2. Ensure heatmap is (H, W, 3)
    if heatmap.ndim != 3 or heatmap.shape[2] != 3:
        raise ValueError(f"Expected heatmap of shape (H, W, 3), got {heatmap.shape}.")

    # 3. Convert gray to BGR by stacking or using cv2.cvtColor
    #    Match types: if gray is uint8, keep uint8; if float, keep float.
    if gray.dtype == np.uint8:
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # If float32/64, cv2.cvtColor might expect values in [0,1]; to be safe, stack manually:
        gray_bgr = np.repeat(gray[:, :, np.newaxis], 3, axis=2)

    # 4. Compute per-pixel alpha from heatmap intensity
    #    First, cast to float for normalization:
    heatmap_f = heatmap.astype(np.float32)

    #    Compute intensity per pixel:
    max_rgb = np.max(heatmap_f, axis=2)    # shape: (H, W)
    gamma = 0.5  # gamma < 1 amplifies lower intensities  
    alpha = np.power(max_rgb / (np.max(max_rgb) + eps), gamma)
    alpha = alpha[:, :, np.newaxis]  # shape: (H, W, 1)

    # 5. Blend: overlay = alpha * heatmap + (1 - alpha) * gray_bgr
    #    If original dtype is uint8, perform blending in float then convert back.
    orig_dtype = heatmap.dtype
    if np.issubdtype(orig_dtype, np.integer):
        # Convert both to float32 in [0,255], blend, then round+astype back to uint8
        heatmap_f_255 = heatmap_f
        gray_bgr_f   = gray_bgr.astype(np.float32)
        overlay_f = alpha * heatmap_f_255 + (1.0 - alpha) * gray_bgr_f
        overlay = np.clip(overlay_f, 0, 255).astype(np.uint8)
    else:
        # Assume float inputs (e.g., float32 in [0,1] or [0,255]); just blend directly
        gray_bgr_f = gray_bgr.astype(np.float32)
        overlay_f = alpha * heatmap_f + (1.0 - alpha) * gray_bgr_f
        overlay = overlay_f.astype(orig_dtype)

    return overlay

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, app_model=None, 
               max_depth=5.0, volume=None, use_depth_filter=False, error_colormap='hot', create_heatmap_mesh=True):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")
    render_error_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_error", error_colormap)

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(render_error_path, exist_ok=True)

    custom_color = error_map_colors(gaussians, colormap_name=error_colormap)

    # Validate the custom_colors tensor
    validate_override_color(custom_color, gaussians)

    depths_tsdf_fusion = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, gt_gray = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        out_error = render(view, gaussians, pipeline, background, override_color=custom_color, app_model=app_model)
        rendering_error = out_error["render"].clamp(0.0, 1.0)

        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if create_heatmap_mesh:
            heatmap_depth = out_error["plane_depth"].squeeze()
            depth_tsdf = heatmap_depth.clone()

        rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        torchvision.utils.save_image(rendering_error[[2, 1, 0], :, :], os.path.join(render_error_path, view.image_name + ".png"))

        gt_gray_np = (gt_gray.permute(1,2,0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        rendering_error_np = (rendering_error.permute(1,2,0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8) #  BGR format 

        # 1. Compute new, half‐resolution dimensions
        original_h, original_w = gt_gray_np.shape[:2] # 1162 x 1554 x 1
        new_h, new_w = original_h // 2, original_w // 2

        # 2. Downsample the gray image (single channel) using INTER_AREA
        gray_single = gt_gray_np[:, :, 0] # 777 x 581
        resized_gray = cv2.resize(gray_single, (new_w, new_h), interpolation=cv2.INTER_AREA) # 777 x 581

        overlay_result = overlay_heatmap(gray=resized_gray, heatmap=rendering_error_np)

        cv2.imwrite(os.path.join(render_error_path, f"{view.image_name}_overlay.jpg"), overlay_result)

        # colormap_small = cv2.resize(rendering_error_np, (new_w, new_h),
        #                             interpolation=cv2.INTER_LINEAR)  # shape: (half_h x half_w x 3)

        # # 3. Convert the resized colormap from BGR → HSV
        # hsv_colormap = cv2.cvtColor(colormap_small, cv2.COLOR_BGR2HSV)  # 777 x 581 x 3
        # V   = hsv_colormap[..., 2].astype(np.float32) / 255.0 

        # # # 4. Expand gray_half into BGR so we can blend
        # gray_bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)  # H₂×W₂×3, uint8

        # # 5. Blend: out = C * α + Gray * (1−α) for each channel
        # alpha = V[..., None]                             # shape = H₂×W₂×1
        # overlay_half = (rendering_error_np.astype(np.float32) * alpha
        #             + gray_bgr.astype(np.float32) * (1.0 - alpha))
        # overlay_half = overlay_half.astype(np.uint8)      # back to uint8



        # # 4. Overwrite the V channel in HSV with the downsampled gray
        # hsv_colormap[..., 2] = resized_gray

        # # 5. Convert from HSV → BGR *directly* (so it’s ready for cv2.imwrite)
        # overlay_bgr = cv2.cvtColor(hsv_colormap, cv2.COLOR_HSV2BGR)

        # 6. overlay = cv2.addWeighted(rendering_np, 0.2, rendering_error_np, 1.0, 0)


        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1,2,0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir*depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0

        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth>max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))

            if create_heatmap_mesh:
                color = o3d.io.read_image(os.path.join(render_error_path, view.image_name + ".png"))
                depth = o3d.geometry.Image((ref_depth*1000).astype(np.uint16))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, max_depth : float,
                voxel_size : float, num_cluster: int, use_depth_filter : bool, error_colormap : str, create_heatmap_mesh : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0*voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        
        # Generate custom colors based on scale in the normal direction
        print(f"color mapping used: {error_colormap}")
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, error_colormap=error_colormap, create_heatmap_mesh=create_heatmap_mesh)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            if create_heatmap_mesh:
                path = os.path.join(dataset.model_path, "mesh_heatmap", error_colormap)
                makedirs(path, exist_ok=True)
                print("creating heatmap mesh:")
            else:
                path = os.path.join(dataset.model_path, "mesh")
                makedirs(path, exist_ok=True)
                print("creating normal mesh:")

            os.makedirs(path, exist_ok=True)
            
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            
            mesh = post_process_mesh(mesh, num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh, 
                                       write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--create_heatmap_mesh", action="store_true")
    parser.add_argument("--error_colormap", type=str, default="hot")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args.error_colormap, args.create_heatmap_mesh)