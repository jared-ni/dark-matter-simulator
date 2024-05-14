#!/usr/bin/env bash
#SBATCH --job-name=gravity
#SBATCH --output=gravity_%j.out
#SBATCH --error=gravity_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

spack load openmpi@4.1.6
spack load papi
export LD_LIBRARY_PATH=/shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/papi-6.0.0.1-jevzvgsbwnpxam5t7obqxl45ztwxwvhi/lib
mpic++ -o gravity gravity.cpp -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -mavx2 -fopenmp -I /shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/papi-6.0.0.1-jevzvgsbwnpxam5t7obqxl45ztwxwvhi/include -L /shared/software/spack/opt/spack/linux-amzn2-skylake_avx512/gcc-7.3.1/papi-6.0.0.1-jevzvgsbwnpxam5t7obqxl45ztwxwvhi/lib -lpapi
srun --export=all ./gravity

exit $?
