#!/usr/bin/env bash

# train & test

mkdir tmp model; cp -r ../run_dict/model/alphabets/ ./model/;

RGPU=1
SEED=1234

echo "Current seed is $SEED"

PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES=$RGPU python2 ../src/examples/GraphParser.py \
--mode FastLSTM \
--no_CoRNN \
--hidden_size 300 \
--num_layers 6 \
--d_k 64 \
--d_v 64 \
--arc_space 512 \
--type_space 128 \
--opt adam \
--decay_rate 0.75 \
--epsilon 1e-4 \
--gamma 0.0 \
--clip 5.0 \
--schedule 20 \
--double_schedule_decay 5 \
--use_warmup_schedule \
--check_dev 5 \
--unk_replace 0.5 \
--freeze \
--pos \
--multi_head_attn \
--num_head 8 \
--word_embedding word2vec \
--word_path './model/alphabets/joint_embed.vec' \
--char_embedding random \
--punctuation 'PUNCT' 'SYM' \
--train "../data2.2_more/en_train.conllu" \
--dev "../data2.2_more/en_dev.conllu" \
--test "../data2.2_more/en_test.conllu" \
--vocab_path './model/' \
--model_path './model/' \
--model_name 'network.pt' \
--p_in 0.2 \
--p_out 0.2 \
--p_rnn 0.2 0.1 0.2 \
--learning_rate 0.0001 \
--num_epochs 1000 \
--trans_hid_size 512 \
--pos_dim 50 \
--char_dim 50 \
--num_filters 50 \
--position_dim 0 \
--enc_clip_dist 10 \
--batch_size 80 \
--seed $SEED

# --char \

#RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh dev biaffine |& tee log_dev
RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh test biaffine |& tee log_test
RGPU=$RGPU bash -v ../src/examples/run_more/run_analyze.sh train biaffine |& tee log_train

#
# b neuronlp2/transformer/multi_head_attn:140
# b neuronlp2/models/parsing:438

# run
# RGPU=2 SEED=1234 bash -v go.sh |& tee log
