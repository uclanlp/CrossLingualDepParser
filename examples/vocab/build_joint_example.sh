#!/usr/bin/env bash

# build
embed_path=./
embed_path1=$embed_path/wiki.multi.en.vec
embed_path2=$embed_path/wiki.multi.de.vec
embed_path3=$embed_path/wiki.multi.fr.vec
data_path=./
data0=$data_path/en-ud-train.conllu
data1=$data_path/en-ud-dev.conllu
data2=$data_path/en-ud-test.conllu
data3=$data_path/de-ud-test.conllu
data4=$data_path/fr-ud-test.conllu

PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES= python2 ../src/examples/vocab/build_joint_vocab_embed.py \
    --embed_paths $embed_path1 $embed_path2 $embed_path3 \
	--embed_lang_ids en de fr \
	--data_paths $data0 $data1 $data2 $data3 $data4 \
	--model_path ./model/
