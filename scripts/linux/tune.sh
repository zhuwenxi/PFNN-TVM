#!/bin/bash
set -e

if [ $# -ne 3 ]
then
    echo "Usage: $0 <mode(autoTVM/ansor)> <interp(A/B)> <use_extern(no_extern/extern)>"
    exit -1
fi

MODE=$1
INTERP=$2
USE_EXTREN=$3


if [ $MODE != 'autoTVM' ] && [ $MODE != 'ansor' ]; then
    echo "error MODE"
    exit
fi
if [ $INTERP != 'A' ] && [ $INTERP != 'B' ]; then
    echo "error INTERP"
    exit
fi
if [ $USE_EXTREN == 'extern' ]; then
    EXTREN="--extern"
elif [ $USE_EXTREN == 'no_extern' ]; then
    EXTREN=""
else
    echo "error USE_EXTREN"
    exit
fi

SHELL_ROOT=$(pwd)
PROJECT_ROOT=$(cd $(dirname $0); pwd)/../..
cd ${PROJECT_ROOT}

if [[ -d "logs" ]]; then
    rm -rf logs
fi
if [ $USE_EXTREN == 'extern' ]; then
    mkdir -p logs/logs_interp${INTERP}_${MODE}_xsmm
else
    mkdir -p logs/logs_interp${INTERP}_${MODE}
fi

python ./python/gen_static_tvminfer_lib.py -p linux -d 256 --mode ${MODE} --interp ${INTERP} ${EXTREN} --tune

mv logs tune_log_$(date "+%Y%m%d%H%M")