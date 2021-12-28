#!/bin/bash --login
#PBS -l select=1:ncpus=${cores}
#PBS -l place=scatter
#PBS -l walltime=${time}
#PBS -A ec105
module load gcc/6.2.0
cd $PBS_O_WORKDIR
% for cmd in commands:
${cmd}
% endfor
