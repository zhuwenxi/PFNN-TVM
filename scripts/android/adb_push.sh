#!/bin/bash
set -e

if [ $# -ne 1 ]
then
    echo "Usage: $0 <file_path_in_device> "
    exit -1
fi

SHELL_ROOT=$(pwd)
PROJECT_ROOT=$(cd $(dirname $0); pwd)/../..
cd ${PROJECT_ROOT}

FILE_PATH=$1

adb push ./build/pfnn_benchmark ${FILE_PATH} # Upload the binary executable file
adb push ./build/cache_block_size.json ${FILE_PATH} # Upload the pre-packed size file
adb push ./models ${FILE_PATH} # Upload the models weights and inputs
