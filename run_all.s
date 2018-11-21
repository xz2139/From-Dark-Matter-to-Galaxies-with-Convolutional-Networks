#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=dark
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1007@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR
module load python3/intel/3.6.3
source /home/yw1007/myenv/bin/activate
modelidx=3
lossweight=1
yweight=1
python src1/main.py --lr 0.0005 --loss_weight $lossweight --model_idx $modelidx --epochs 15 --target_cat 'count' --target_class 0 --load_model 0 \
--save_name "model_full_${modelidx}_${lossweight}_${yweight}" --record_results 0 --yfloss_weight $yweight > result_full_${modelidx}_${lossweight}_${yweight}.txt
