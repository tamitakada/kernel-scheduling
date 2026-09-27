module load cuda/13.1
source venv/bin/activate

rm cupti_sched.so
export CUDA_ROOT=/modules/opt/linux-ubuntu24.04-x86_64/nvhpc/Linux_x86_64/24.9/cuda/13.1
gcc -shared -fPIC -o cupti_sched.so cupti_sched.c \
    -I$CUDA_ROOT/include \
    -I$CUDA_ROOT/extras/CUPTI/include \
    -L$CUDA_ROOT/extras/CUPTI/lib64 \
    -lcupti -lcuda
export LD_LIBRARY_PATH=$CUDA_ROOT/extras/CUPTI/lib64:$LD_LIBRARY_PATH
