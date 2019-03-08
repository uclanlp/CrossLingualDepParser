#!/usr/bin/env bash

# RGPU=$RGPU bash run5.sh ?.sh

BASE_DIR=`pwd`
SCRIPT_FNAME=$1
SEED_BASE=123

for i in 1 2 3 4 5; do

CUR_DIR="$BASE_DIR/${SCRIPT_FNAME}_${i}"

if [ ! -d "$CUR_DIR" ]; then
    mkdir $BASE_DIR/${SCRIPT_FNAME}_${i};
    cd $BASE_DIR/${SCRIPT_FNAME}_${i};
    cp ../$SCRIPT_FNAME go.sh;
    RGPU=$RGPU SEED="${SEED_BASE}$i" bash -v go.sh |& tee log;
fi

done
