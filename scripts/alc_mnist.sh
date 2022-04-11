#!/usr/bin/zsh
#SBATCH --job-name=alc-research
#SBATCH --mem=16gb
#SBATCH --ntasks=1
#SBATCH --array=1-5%5
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --output=/mnt/stud/home/phahn/alc-research/logs/%A_%x_%a.log
#SBATCH --exclude=vana
date;hostname;pwd
source /mnt/stud/home/phahn/.zshrc

conda activate /mnt/stud/home/phahn/.conda/envs/alc

# This is a script to execute an active learning cycle simulation with the possibility of calibration.
DIRECTORY=`pwd`
cd $DIRECTORY
export PYTHONPATH=$PYTHONPATH:$DIRECTORY


srun python /mnt/stud/home/phahn/alc-research/codeV2/main.py \
    --activation_function "relu" \
	--batch_size 200 \
	--cal_data_factor 0.3 \
	--calibrator "histogram" \
	--criterion "cel" \
	--dropout_rate 0.25 \
	--evaluation_mode "single" \
	--hidden_layer_size 150 \
	--input_size 784 \
	--learning_rate 0.001 \
	--num_alc 18 \
	--num_bins 5 \
	--num_classes 10 \
	--num_epochs 100 \
	--num_labeld_samples 50 \
	--num_mc_passes 10 \
	--num_purchases 5 \
	--num_samples 10000 \
	--num_test_samples 10000 \
	--optimizer "Adam" \
	--output_size 10 \
	--path_to_data "./" \
	--query_strategy "entropy" \
	--random_seed $SLURM_ARRAY_TASK_ID \
	--test_batch_size 32 \
	--train_batch_size 32 