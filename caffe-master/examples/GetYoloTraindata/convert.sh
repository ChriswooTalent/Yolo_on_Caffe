#!/usr/bin/env sh

CAFFE_ROOT=D:/Code_local/caffe/caffe-master/caffe-master
CAFFEDATA_ROOT=D:/Code_local/caffe/caffe_yolo
ROOT_DIR=D:/VOCdevkit/VOC2012/
LABEL_FILE=$CAFFEDATA_ROOT/label_map.txt

# 2007 + 2012 trainval
LIST_FILE=$CAFFEDATA_ROOT/trainval.txt
LMDB_DIR=./lmdb/trainval_lmdb
SHUFFLE=true

# 2012 test
#TESTLIST_FILE=$CAFFEDATA_ROOT/test_2007.txt
#TESTLMDB_DIR=./lmdb/test2007_lmdb
#SHUFFLE=false

RESIZE_W=448
RESIZE_H=448

$CAFFE_ROOT/Build/x64/Debug/convert_box_data.exe --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $ROOT_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

