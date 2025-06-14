import os

# scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scenes = [24]
data_base_path='data/dtu_dataset/dtu'
out_base_path='output/dtu'
eval_path='data/dtu_dataset/dtu_eval'
out_name='gim_roma_15000_0.6_dyna'
gpu_id=0

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "-r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train_gim_roma.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --error_colormap hot"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python new_render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)