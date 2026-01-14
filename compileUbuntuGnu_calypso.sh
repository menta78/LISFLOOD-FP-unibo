#module load gcc-12.2.0/12.2.0
export NUMADIR=/home/dosreislopes/usr/numactl-2.0.15_gcc
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NUMADIR/lib
CONFIG=config/sharc-gcc-cpu-calypso  make
#mv lisflood lisflood_debug
