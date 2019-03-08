#!/usr/bin/env bash

# test

RGPU=1      # set up your GPU-ID

#RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh dev biaffine |& tee log_dev
RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh test biaffine |& tee log_test
#RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh train biaffine |& tee log_train

#
# b neuronlp2/transformer/multi_head_attn:140
# b neuronlp2/models/parsing:438

# run
# RGPU=2 SEED=1234 bash -v go.sh |& tee log
