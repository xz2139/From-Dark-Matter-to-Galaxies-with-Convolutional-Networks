#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=dark
#SBATCH --mail-type=END
#SBATCH --mail-user=bob.smith@nyu.edu
#SBATCH --output=slurm_%j.out
  
#module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR
python src/main.py --lr 0.00001 --loss_weight 80 --model_idx 2 --epochs 20 --target_cat 'count' --target_class 0 --load_model 0 \
--conv1_out 52 --conv3_out 60 --conv5_out 68  > result_inception_all.txt