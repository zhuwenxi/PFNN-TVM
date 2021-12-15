#!/bin/bash
set -e

if [ $# -ne 4 ]
then
    echo "Usage: $0 <mode(manual/ansor)> <interp(A/B)> <use_extern(no_extern/extern)> <logs_dir> "
    exit -1
fi

SHELL_ROOT=$(pwd)
PROJECT_ROOT=$(cd $(dirname $0); pwd)/../..
cd ${PROJECT_ROOT}

MODE=$1
INTERP=$2
USE_EXTREN=$3
LOGS=${SHELL_ROOT}/$4

if [ $MODE != 'manual' ] && [ $MODE != 'ansor' ]; then
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

if [ $MODE != 'manual' ]; then
    if [[ -d "logs" ]]; then
        rm -rf logs
    fi
    mkdir -p logs
    if [ $USE_EXTREN == 'extern' ] && [ -d $LOGS/logs_interp${INTERP}_${MODE}_xsmm ]; then
        cp -r $LOGS/logs_interp${INTERP}_${MODE}_xsmm logs
    elif [ $USE_EXTREN == 'no_extern' ] && [ -d $LOGS/logs_interp${INTERP}_${MODE} ]; then
        cp -r $LOGS/logs_interp${INTERP}_${MODE} logs
    else
        echo "error LOGS_DIR"
        exit
    fi
fi

# generate models data
if [[ -d "models" ]]; then
    rm -rf models
fi
mkdir -p models
python ./python/gen_model_data.py
echo "gen_model_data Done."

if [[ -d "build" ]]; then
    rm -rf build
fi
mkdir -p build

# generate tvminfer scheduler lib

python ./python/gen_static_tvminfer_lib.py -p linux -d 256 --mode ${MODE} --interp ${INTERP} ${EXTREN}
mv libtvminfer_linux_x86_64.a build/
mv dummy_stub.cc build/
echo "gen_linux_libs_static Done."

python ./python/gen_cache_block_json.py -d 256 --mode ${MODE} --interp ${INTERP} ${EXTREN} 
mv cache_block_size.json build/
echo "gen_cache_block_json Done."

cd build
cmake -DPLATFORM=linux \
      -DEIGEN_INTERP=A \
      -DTVM_INTERP=${INTERP} \
      ..

make 
