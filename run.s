#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=40GB
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
python src/main.py --medium1 1 --lr 0.00001 --loss_weight 30 --model_idx 2 --epochs 6 --target_cat 'count' --target_class 0 --load_model 0 \
--conv1_out 52 --conv3_out 60 --conv5_out 68 --save_name 'yqloss10' --record_results 0 --yqloss_weight 1 > result_yqloss.txt