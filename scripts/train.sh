# basic setting
basic_dir=`pwd`
accelerate=/home/share/caoxu/miniconda3/envs/python310/bin/accelerate
accelerate_config=${basic_dir}/../conf/accelerate.yaml
# diy config
task=pt
pipeline_name=continue
pipeline_config=${basic_dir}/../conf/${task}/${pipeline_name}.yaml
# run script
cd ../src
${accelerate} launch --num_cpu_threads_per_process 2 \
                     --config_file ${accelerate_config} \
                     main.py --pipeline_config ${pipeline_config}