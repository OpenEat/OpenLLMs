basic_dir=`pwd`

accelerate=/root/autodl-tmp/miniconda3/envs/python10/bin/accelerate
accelerate_config=${basic_dir}/../conf/accelerate.yaml
pipeline_config=${basic_dir}/../conf/pipeline.yaml
cd ../src
CUDA_VISIBLE_DEVICES=0,1 ${accelerate} launch --num_cpu_threads_per_process 2 --multi_gpu \
                     --config_file ${accelerate_config} \
                     main.py --pipeline_config ${pipeline_config}