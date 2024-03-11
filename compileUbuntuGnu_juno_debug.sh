module load gcc-12.2.0/12.2.0
export NUMADIR=/users_home/opa/id22022/usr/numactl-2.0.15_gcc
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMADIR/lib
CONFIG=config/sharc-gcc-cpu-juno-debug  make
mv lisflood lisflood_debug
