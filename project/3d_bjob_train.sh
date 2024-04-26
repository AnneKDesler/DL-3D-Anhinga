#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_8
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s185231@dtu.dk
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -R "select[sxm2]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o 3D_batch_output/train_%J.out
#BSUB -e 3D_batch_output/train_%J.err
# -- end of LSF options --

source /dtu/3d-imaging-center/courses/conda/conda_init.sh
conda activate env-02510

export CUDA_VISIBLE_DEVICES=0

learning_rates=(1e-3 5e-4 1e-4 1e-5)
batch_sizes=(8 16 32)

for n in "${learning_rates[@]}"
do
    for m in "${batch_sizes[@]}"
    do
        python -u 3d_train.py --lr $n --batch_size $m --num_epochs 100
    done
done

#python -u train_prof.py
#python -u train_prof_01.py
#python -u train_prof_02.py
#python -u train_prof_03.py
#python -u train_prof_04.py
