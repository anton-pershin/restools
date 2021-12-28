#$ -cwd -V
#$ -l h_rt=${time}
#$ -pe ib ${cores}
module switch intel/19.0.4 gnu/native
module load openmpi/3.1.4
module load fftw/3.3.8
module load netcdf/4.6.3

% for cmd in commands:
mpirun ${cmd}
% endfor