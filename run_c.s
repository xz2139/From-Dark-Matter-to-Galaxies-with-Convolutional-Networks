#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # p40 k80 p100 1080
#SBATCH --time=18:00:00
#SBATCH --mem=80GB
#SBATCH --job-name=2
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
modelidx=2
target_class=0
vel=0
#lossweight=160
normalize=0
conv1_out=6
conv3_out=8
conv5_out=10

modelname=$modelidx
if [ $modelidx -eq 2 ] || [ $modelidx -eq 7 ]
then
        modelname="${modelname}@${conv1_out}@${conv3_out}@${conv5_out}"
fi
for lossweight in 40 80 160
do
name="${modelname}_${lossweight}_v${vel}_r0"
save_name="model_${name}"

python src/main.py --lr 0.001 --loss_weight $lossweight --model_idx $modelidx --epochs 22 --target_cat 'count' --target_class ${target_class} --load_model 0 \
--save_name $save_name --normalize $normalize  --record_results 0 --vel $vel > result_full_${name}.txt

done
