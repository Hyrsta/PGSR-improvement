import os

data_base_path='data/droneV3'
out_base_path='output/droneV3'
out_name='roma_30000_0.6_dyna'
gpu_id=0

#     common_args = "--quiet -r2 --ncc_scale 0.5"
# common_args = "-r2 --ncc_scale 0.5"
# cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path} -m {out_base_path}/{out_name} {common_args}'
# print(cmd)
# os.system(cmd)

common_args = f"--quiet --num_cluster 1 --voxel_size 0.01 --max_depth 5.0 --error_colormap hot"
cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python new_render.py -m {out_base_path}/{out_name} {common_args}'
print(cmd)
os.system(cmd)