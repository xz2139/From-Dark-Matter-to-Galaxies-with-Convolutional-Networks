#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=dark
#SBATCH --mail-type=END
#SBATCH --mail-user=ys3202@nyu.edu
#SBATCH --output=slurm_%j.out
  
#module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR
# python src/main.py --medium 1 --lr 0.00001 --loss_weight 20 --model_idx 3 --epochs 8 --target_cat 'count' --target_class 0 --load_model 0 \
# --conv1_out 52 --conv3_out 60 --conv5_out 68 --save_name 'yqloss_20_0' --record_results 0 --yfloss_weight 0 > result_3_20_0.txt
modelidx=5
lossweight=5
yweight=0
python src/main.py --medium1 1 --lr 0.0001 --loss_weight $lossweight --model_idx $modelidx --epochs 4 --target_cat 'count' --target_class 0 --load_model 0 \
--save_name "" --record_results 0 --yfloss_weight $yweight > result_${modelidx}_${lossweight}_${yweight}.txt
#model_${modelidx}_${lossweight}_${yweight}