#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=24:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=dark
#SBATCH --mail-type=END
#SBATCH --mail-user=ys3202@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge
#module load tensorflow/python3.5/1.4.0 
#imodule load cudnn/8.0v6.0
#module load cuda/8.0.44
#RUNDIR=$home/ys3202/dark/run-${SLURM_JOB_ID/.*}
#mkdir -p $RUNDIR
#module load python3/intel/3.6.3
#source /home/yw1007/myenv/bin/activate

# model_idx: 
# - 4 One layer convolution + R2Unet
# - 5 R2Unet + R2Unet
# - 7 inception + R2Unet

# - C_model: C model name used as the first-phase model
# - loss_weight: weight for the loss function
# - normalize: should be the same as C model
# - vel: should be the same as C model


modelidx=7
target_class=1
# keep vel and normalize the same as the C model
vel=0
normalize=0
#remember to remove .pth
C_model="model_2@6@8@10_40_v0_n0"

for lossweight in 0.6 0.8 1 1.5
do

name="${modelidx}_${lossweight}@${C_model}"
save_name="model_${name}"

python src/main.py --lr 0.001 --loss_weight $lossweight --model_idx $modelidx --epochs 20 --target_cat 'count' --target_class ${target_class} --load_model 0 \
--save_name $save_name --normalize $normalize  --record_results 0 --vel $vel --C_model $C_model > result_full_${name}.txt

done