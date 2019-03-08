#!/usr/bin/env bash

# build
embed_path=../data2.2_more/
embed_path1=$embed_path/wiki.multi.en.vec
data_path=../data2.2_more/
data0=$data_path/en_train.conllu
data1=$data_path/en_dev.conllu
data2=$data_path/en_test.conllu

PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES= python2 ../src/examples/vocab/build_joint_vocab_embed.py \
--embed_paths $embed_path1 \
--embed_lang_ids en \
--data_paths $data0 $data1 $data2 \
--model_path ./model/
