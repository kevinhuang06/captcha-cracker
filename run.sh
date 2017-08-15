#!/usr/bin/env sh

# 1. modify config.py at current dir

# 2. gene hdf5 
cp /data1/kevin.H/base_model/preprocess.py ./
if [ "$2" != "skip" ];then
    python preprocess.py 
# 3. modify input file of prototxt
    if [ "$1" = "small" ]; then  
        cp  ../../base_model/small_net/*prototxt ./
        echo 'use small net'
    else
        cp  ../../base_model/*prototxt ./  
        echo 'use regular net'
    fi  
    mkdir -p model
    sed -i 's/TRAIN-DATA/hdf5\/train_hdf5_list.txt/g' validnet_train_test.prototxt
    sed -i 's/VALIDATION-DATA/hdf5\/test_hdf5_list.txt/g' validnet_train_test.prototxt
fi
# 4 train 
/home/xdyang/caffe/build/tools/caffe train \
      --solver=validnet_solver.prototxt > xxxxx.log


