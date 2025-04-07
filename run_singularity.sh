#!/bin/bash
#SBATCH --job-name=person-reid
#SBATCH --nodelist=nv172
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=peta_modi.out
#SBATCH --error=peta_modi.err


################ Number of total process ##########################

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


echo "Run started at:- "
date

#cd /purestorage/AILAB/AI_2/youhans/workspace/reid/person/HumanBench/PATH
cd /purestorage/AILAB/AI_2/youhans/workspace/reid/person/HumanBench/UniHCP

echo Start

srun nvidia-smi

###### PATH

#srun python multitask.py --expname pa100k_vitbase_modi \
#  --config experiments/L2_full_setting_joint_v100_32g/finetune/attr/pa100k_vitbase_SGD_lr1e2x1_stepLRx2_wd5e4_backboneclip_dpr03_30ep.yaml

#srun python multitask.py --expname peta_vitbase_modi \
#  --config experiments/L2_full_setting_joint_v100_32g/finetune/attr/peta_vitbase_SGD_lr1e2x1_stepLRx2_wd1e4_dpr01_80ep.yaml

############ Test

#srun python test.py --expname test_pa100k_vitbase_modi \
#  --config experiments/L2_full_setting_joint_v100_32g/finetune/attr/pa100k_vitbase_SGD_lr1e2x1_stepLRx2_wd5e4_backboneclip_dpr03_30ep.yaml \
#  --test_config experiments/L2_full_setting_joint_v100_32g/finetune/attr/pedattr_pa100k_test.yaml \
#  --spec_ginfo_index 0\
#  --load-path outputs/checkpoints/pa100k_vitbase_modi/pa100k_vitbase_modi.pth

#srun python test.py --expname test_peta_vitbase_modi \
#  --config experiments/L2_full_setting_joint_v100_32g/finetune/attr/peta_vitbase_SGD_lr1e2x1_stepLRx2_wd1e4_dpr01_80ep.yaml \
#  --test_config experiments/L2_full_setting_joint_v100_32g/finetune/attr/pedattr_peta_test.yaml \
#  --spec_ginfo_index 0\
#  --load-path outputs/checkpoints/peta_vitbase_modi/peta_vitbase_modi.pth



###### UniHCP

#srun python multitask.py --expname pa100k_vitbase_modi \
#  --config experiments/unihcp/release/pa100k_vitbase_coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256.yaml \
#  --tcp_port 4672


#srun python multitask.py --expname peta_vitbase_modi \
#  --config experiments/unihcp/release/peta_vitbase_coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256.yaml \
#  --tcp_port 4672


############ Test

#srun python test.py --expname test_pa100k_vitbase_modi \
#  --config experiments/unihcp/release/pa100k_vitbase_coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256.yaml \
#  --test_config experiments/unihcp/release/vd_pa100k_lpe_test.yaml \
#  --spec_ginfo_index 0\
#  --load-path checkpoints/pa100k_vitbase_modi/pa100k_vitbase_modi.pth

srun python test.py --expname test_peta_vitbase_modi \
  --config experiments/unihcp/release/peta_vitbase_coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256.yaml \
  --test_config experiments/unihcp/release/vd_peta_lpe_test.yaml \
  --spec_ginfo_index 0\
  --load-path checkpoints/peta_vitbase_modi/peta_vitbase_modi.pth

echo Done
