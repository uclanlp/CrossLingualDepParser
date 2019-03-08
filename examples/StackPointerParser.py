from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam, SGD, Adamax
from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.models import StackPtrNet
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

from neuronlp2.io_multi import guess_language_id, create_alphabets

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducibility')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'], help='architecture of rnn',
                             required=True)
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--decoder_input_size', type=int, default=256,
                             help='Number of input units in decoder RNN.')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--encoder_layers', type=int, default=1, help='Number of layers of encoder RNN')
    args_parser.add_argument('--decoder_layers', type=int, default=1, help='Number of layers of decoder RNN')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--trans_hid_size', type=int, default=1024,
                             help='#hidden units in point-wise feed-forward in transformer')
    args_parser.add_argument('--d_k', type=int, default=64, help='d_k for multi-head-attention in transformer encoder')
    args_parser.add_argument('--d_v', type=int, default=64, help='d_v for multi-head-attention in transformer encoder')
    args_parser.add_argument('--multi_head_attn', action='store_true', help='use multi-head-attention.')
    args_parser.add_argument('--num_head', type=int, default=8, help='Value of h in multi-head attention')
    args_parser.add_argument('--pool_type', default='mean', choices=['max', 'mean', 'weight'],
                             help='pool type to form fixed length vector from word embeddings')
    args_parser.add_argument('--train_position', action='store_true', help='train positional encoding for transformer.')
    args_parser.add_argument('--no_word', action='store_true', help='do not use word embedding.')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--no_CoRNN', action='store_true', help='do not use context RNN.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--coverage', type=float, default=0.0, help='weight for coverage loss')
    args_parser.add_argument('--p_rnn', nargs='+', type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_parser.add_argument('--label_smooth', type=float, default=1.0, help='weight of label smoothing method')
    args_parser.add_argument('--skipConnect', action='store_true', help='use skip connection for decoder RNN.')
    args_parser.add_argument('--grandPar', action='store_true', help='use grand parent.')
    args_parser.add_argument('--sibling', action='store_true', help='use sibling.')
    args_parser.add_argument('--prior_order', choices=['inside_out', 'left2right', 'deep_first', 'shallow_first'],
                             help='prior order of children.', required=True)
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                             required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--vocab_path', help='path for prebuilt alphabets.', default=None)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    args_parser.add_argument('--position_embed_num', type=int, default=200,
                             help='Minimum value of position embedding num, which usually is max-sent-length.')
    args_parser.add_argument('--num_epochs', type=int, default=2000, help='Number of training epochs')

    # lrate schedule with warmup in the first iter.
    args_parser.add_argument('--use_warmup_schedule', action='store_true', help="Use warmup lrate schedule.")
    args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
    args_parser.add_argument('--max_decay', type=int, default=9, help='Number of decays before stop')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--double_schedule_decay', type=int, default=5, help='Number of decays to double schedule')
    args_parser.add_argument('--check_dev', type=int, default=5,
                             help='Check development performance in every n\'th iteration')
    #
    # about decoder's bi-attention scoring with features (default is not using any)
    args_parser.add_argument('--dec_max_dist', type=int, default=0,
                             help="The clamp range of decoder's distance feature, 0 means turning off.")
    args_parser.add_argument('--dec_dim_feature', type=int, default=10, help="Dim for feature embed.")
    args_parser.add_argument('--dec_use_neg_dist', action='store_true',
                             help="Use negative distance for dec's distance feature.")
    args_parser.add_argument('--dec_use_encoder_pos', action='store_true',
                             help="Use pos feature combined with distance feature for child nodes.")
    args_parser.add_argument('--dec_use_decoder_pos', action='store_true',
                             help="Use pos feature combined with distance feature for head nodes.")
    args_parser.add_argument('--dec_drop_f_embed', type=float, default=0.2, help="Dropout for dec feature embeddings.")
    #
    # about relation-aware self attention for the transformer encoder (default is not using any)
    # args_parser.add_argument('--rel_aware', action='store_true',
    #                          help="Enable relation-aware self-attention (multi_head_attn flag needs to be set).")
    args_parser.add_argument('--enc_use_neg_dist', action='store_true',
                             help="Use negative distance for enc's relational-distance embedding.")
    args_parser.add_argument('--enc_clip_dist', type=int, default=0,
                             help="The clipping distance for relative position features.")
    #
    # other options about how to combine multiple input features (have to make some dims fit if not concat)
    args_parser.add_argument('--input_concat_embeds', action='store_true', help="Concat input embeddings, otherwise add.")
    args_parser.add_argument('--input_concat_position', action='store_true', help="Concat position embeddings, otherwise add.")
    args_parser.add_argument('--position_dim', type=int, default=300, help='Dimension of Position embeddings.')
    #
    args_parser.add_argument('--train_len_thresh', type=int, default=100, help='In training, discard sentences longer than this.')

    args = args_parser.parse_args()

    # =====
    # fix data-prepare seed
    random.seed(1234)
    np.random.seed(1234)
    # model's seed
    torch.manual_seed(args.seed)

    # =====

    # if output directory doesn't exist, create it
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    logger = get_logger("PtrParser", args.model_path + 'log.txt')

    logger.info('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
    logger.info('{0}\n'.format(args))

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    vocab_path = args.vocab_path if args.vocab_path is not None else args.model_path
    model_path = args.model_path
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    input_size_decoder = args.decoder_input_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    encoder_layers = args.encoder_layers
    decoder_layers = args.decoder_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    cov = args.coverage
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    label_smooth = args.label_smooth
    unk_replace = args.unk_replace
    prior_order = args.prior_order
    skipConnect = args.skipConnect
    grandPar = args.grandPar
    sibling = args.sibling
    beam = args.beam
    punctuation = args.punctuation

    freeze = args.freeze
    use_word_emb = not args.no_word
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    use_con_rnn = not args.no_CoRNN

    use_pos = args.pos
    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path) if use_word_emb else (None, 0)
    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(vocab_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)

    # todo(warn): should build vocabs previously
    assert os.path.isdir(alphabet_path), "should have build vocabs previously"
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_stacked_data.create_alphabets(
        alphabet_path, train_path, data_paths=[dev_path, test_path], max_vocabulary_size=50000, embedd_dict=word_dict)
    # word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = create_alphabets(alphabet_path,
    #     train_path, data_paths=[dev_path, test_path], max_vocabulary_size=50000, embedd_dict=word_dict)
    max_sent_length = max(max_sent_length, args.position_embed_num)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    # ===== the reading
    def _read_one(path, is_train):
        lang_id = guess_language_id(path)
        logger.info("Reading: guess that the language of file %s is %s." % (path, lang_id))
        one_data = conllx_stacked_data.read_stacked_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet,
                        type_alphabet, use_gpu=use_gpu, volatile=(not is_train), prior_order=prior_order,
                        lang_id=lang_id, len_thresh=(args.train_len_thresh if is_train else 100000))
        return one_data

    data_train = _read_one(train_path, True)
    num_data = sum(data_train[1])

    data_dev = _read_one(dev_path, False)
    data_test = _read_one(test_path, False)
    # =====

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(
            np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale,
                                                                                                        [1,
                                                                                                         word_dim]).astype(
                    np.float32)
                oov += 1
            table[index, :] = embedding
        logger.info('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_stacked_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        logger.info('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table() if use_word_emb else None
    char_table = construct_char_embedding_table()

    window = 3
    network = StackPtrNet(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                          mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                          num_types, arc_space, type_space, args.pool_type, args.multi_head_attn, args.num_head,
                          max_sent_length, args.trans_hid_size, args.d_k, args.d_v,
                          train_position=args.train_position, embedd_word=word_table, embedd_char=char_table, p_in=p_in,
                          p_out=p_out, p_rnn=p_rnn, biaffine=True, use_word_emb=use_word_emb, pos=use_pos,
                          char=use_char, prior_order=prior_order, use_con_rnn=use_con_rnn, skipConnect=skipConnect,
                          grandPar=grandPar, sibling=sibling, use_gpu=use_gpu,
                          dec_max_dist=args.dec_max_dist, dec_use_neg_dist=args.dec_use_neg_dist,
                          dec_use_encoder_pos=args.dec_use_encoder_pos, dec_use_decoder_pos=args.dec_use_decoder_pos,
                          dec_dim_feature=args.dec_dim_feature, dec_drop_f_embed=args.dec_drop_f_embed,
                          enc_clip_dist=args.enc_clip_dist, enc_use_neg_dist=args.enc_use_neg_dist,
                          input_concat_embeds=args.input_concat_embeds, input_concat_position=args.input_concat_position,
                          position_dim=args.position_dim)

    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, input_size_decoder, hidden_size, encoder_layers, decoder_layers,
                     num_types, arc_space, type_space, args.pool_type, args.multi_head_attn, args.num_head,
                     max_sent_length, args.trans_hid_size, args.d_k, args.d_v]
        kwargs = {'train_position': args.train_position, 'use_word_emb': use_word_emb, 'use_con_rnn': use_con_rnn,
                  'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char,
                  'prior_order': prior_order, 'skipConnect': skipConnect, 'grandPar': grandPar, 'sibling': sibling,
                  'dec_max_dist': args.dec_max_dist, 'dec_use_neg_dist': args.dec_use_neg_dist,
                  'dec_use_encoder_pos': args.dec_use_encoder_pos, 'dec_use_decoder_pos': args.dec_use_decoder_pos,
                  'dec_dim_feature': args.dec_dim_feature, 'dec_drop_f_embed': args.dec_drop_f_embed,
                  'enc_clip_dist': args.enc_clip_dist, 'enc_use_neg_dist': args.enc_use_neg_dist,
                  'input_concat_embeds': args.input_concat_embeds, 'input_concat_position': args.input_concat_position, 'position_dim': args.position_dim}
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if use_word_emb and freeze:
        network.word_embedd.freeze()

    if use_gpu:
        network.cuda()

    save_args()

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    def generate_optimizer(opt, lr, params):
        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return Adam(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        elif opt == 'sgd':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        elif opt == 'adamax':
            return Adamax(params, lr=lr, betas=betas, weight_decay=gamma, eps=eps)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    lr = learning_rate
    optim = generate_optimizer(opt, lr, network.parameters())
    opt_info = 'opt: %s, ' % opt
    if opt == 'adam':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)
    elif opt == 'sgd':
        opt_info += 'momentum=%.2f' % momentum
    elif opt == 'adamax':
        opt_info += 'betas=%s, eps=%.1e' % (betas, eps)

    word_status = 'frozen' if freeze else 'fine tune'
    char_status = 'enabled' if use_char else 'disabled'
    pos_status = 'enabled' if use_pos else 'disabled'
    logger.info("Embedding dim: word=%d (%s), char=%d (%s), pos=%d (%s)" % (
        word_dim, word_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=(%d, %d), input_dec=%d, hidden=%d, arc_space=%d, type_space=%d" % (
        mode, encoder_layers, decoder_layers, input_size_decoder, hidden_size, arc_space, type_space))
    logger.info("train: cov: %.1f, (#data: %d, batch: %d, clip: %.2f, label_smooth: %.2f, unk_repl: %.2f)" % (
        cov, num_data, batch_size, clip, label_smooth, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info('prior order: %s, grand parent: %s, sibling: %s, ' % (prior_order, grandPar, sibling))
    logger.info('skip connect: %s, beam: %d' % (skipConnect, beam))
    logger.info(opt_info)

    num_batches = num_data / batch_size + 1
    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucomlpete_match = 0.0
    dev_lcomplete_match = 0.0

    dev_ucorrect_nopunc = 0.0
    dev_lcorrect_nopunc = 0.0
    dev_ucomlpete_match_nopunc = 0.0
    dev_lcomplete_match_nopunc = 0.0
    dev_root_correct = 0.0

    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete_match = 0.0
    test_lcomplete_match = 0.0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_match_nopunc = 0.0
    test_lcomplete_match_nopunc = 0.0
    test_root_correct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_total_inst = 0
    test_total_root = 0

    # lrate decay
    patient = 0
    decay = 0
    max_decay = args.max_decay
    double_schedule_decay = args.double_schedule_decay

    # lrate schedule
    step_num = 0
    use_warmup_schedule = args.use_warmup_schedule
    warmup_factor = (lr + 0.) / num_batches

    if use_warmup_schedule:
        logger.info("Use warmup lrate for the first epoch, from 0 up to %s." % (lr,))
    #
    for epoch in range(1, num_epochs + 1):
        logger.info('Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f '
                    '(schedule=%d, patient=%d, decay=%d (%d, %d))): ' % (epoch, mode, opt, lr, eps, decay_rate,
                                                                         schedule, patient, decay, max_decay,
                                                                         double_schedule_decay))
        train_err_arc_leaf = 0.
        train_err_arc_non_leaf = 0.
        train_err_type_leaf = 0.
        train_err_type_non_leaf = 0.
        train_err_cov = 0.
        train_total_leaf = 0.
        train_total_non_leaf = 0.
        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            # lrate schedule (before each step)
            step_num += 1
            if use_warmup_schedule and epoch <= 1:
                cur_lrate = warmup_factor * step_num
                # set lr
                for param_group in optim.param_groups:
                    param_group['lr'] = cur_lrate

            # train
            input_encoder, input_decoder = conllx_stacked_data.get_batch_stacked_variable(data_train, batch_size,
                                                                                          unk_replace=unk_replace)
            word, char, pos, heads, types, masks_e, lengths_e = input_encoder
            stacked_heads, children, sibling, stacked_types, skip_connect, masks_d, lengths_d = input_decoder

            optim.zero_grad()
            loss_arc_leaf, loss_arc_non_leaf, \
            loss_type_leaf, loss_type_non_leaf, \
            loss_cov, num_leaf, num_non_leaf = network.loss(word, char, pos, heads, stacked_heads, children, sibling,
                                                            stacked_types, label_smooth,
                                                            skip_connect=skip_connect, mask_e=masks_e,
                                                            length_e=lengths_e, mask_d=masks_d, length_d=lengths_d)
            loss_arc = loss_arc_leaf + loss_arc_non_leaf
            loss_type = loss_type_leaf + loss_type_non_leaf
            loss = loss_arc + loss_type + cov * loss_cov
            loss.backward()
            clip_grad_norm(network.parameters(), clip)
            optim.step()

            num_leaf = num_leaf.data[0]
            num_non_leaf = num_non_leaf.data[0]

            train_err_arc_leaf += loss_arc_leaf.data[0] * num_leaf
            train_err_arc_non_leaf += loss_arc_non_leaf.data[0] * num_non_leaf

            train_err_type_leaf += loss_type_leaf.data[0] * num_leaf
            train_err_type_non_leaf += loss_type_non_leaf.data[0] * num_non_leaf

            train_err_cov += loss_cov.data[0] * (num_leaf + num_non_leaf)

            train_total_leaf += num_leaf
            train_total_non_leaf += num_non_leaf

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 10 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                err_arc_leaf = train_err_arc_leaf / train_total_leaf
                err_arc_non_leaf = train_err_arc_non_leaf / train_total_non_leaf
                err_arc = err_arc_leaf + err_arc_non_leaf

                err_type_leaf = train_err_type_leaf / train_total_leaf
                err_type_non_leaf = train_err_type_non_leaf / train_total_non_leaf
                err_type = err_type_leaf + err_type_non_leaf

                err_cov = train_err_cov / (train_total_leaf + train_total_non_leaf)

                err = err_arc + err_type + cov * err_cov
                log_info = 'train: %d/%d loss (leaf, non_leaf): %.4f, arc: %.4f (%.4f, %.4f), type: %.4f (%.4f, %.4f), coverage: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, err, err_arc, err_arc_leaf, err_arc_non_leaf, err_type, err_type_leaf,
                    err_type_non_leaf, err_cov, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        err_arc_leaf = train_err_arc_leaf / train_total_leaf
        err_arc_non_leaf = train_err_arc_non_leaf / train_total_non_leaf
        err_arc = err_arc_leaf + err_arc_non_leaf

        err_type_leaf = train_err_type_leaf / train_total_leaf
        err_type_non_leaf = train_err_type_non_leaf / train_total_non_leaf
        err_type = err_type_leaf + err_type_non_leaf

        err_cov = train_err_cov / (train_total_leaf + train_total_non_leaf)

        err = err_arc + err_type + cov * err_cov
        logger.info(
            'train: %d loss (leaf, non_leaf): %.4f, arc: %.4f (%.4f, %.4f), type: %.4f (%.4f, %.4f), coverage: %.4f, time: %.2fs' % (
                num_batches, err, err_arc, err_arc_leaf, err_arc_non_leaf, err_type, err_type_leaf, err_type_non_leaf,
                err_cov, time.time() - start_time))

        ################################################################################################
        if epoch % args.check_dev != 0:
            continue

        # evaluate performance on dev data
        network.eval()
        pred_filename = 'tmp/%spred_dev%d' % (str(uid), epoch)
        pred_writer.start(pred_filename)
        gold_filename = 'tmp/%sgold_dev%d' % (str(uid), epoch)
        gold_writer.start(gold_filename)

        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_total = 0
        dev_ucomlpete = 0.0
        dev_lcomplete = 0.0
        dev_ucorr_nopunc = 0.0
        dev_lcorr_nopunc = 0.0
        dev_total_nopunc = 0
        dev_ucomlpete_nopunc = 0.0
        dev_lcomplete_nopunc = 0.0
        dev_root_corr = 0.0
        dev_total_root = 0.0
        dev_total_inst = 0.0
        for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_dev, batch_size):
            input_encoder, _ = batch
            word, char, pos, heads, types, masks, lengths = input_encoder
            heads_pred, types_pred, _, _ = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam,
                                                          leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            heads = heads.data.cpu().numpy()
            types = types.data.cpu().numpy()

            pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
            gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

            stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                    word_alphabet, pos_alphabet, lengths,
                                                                    punct_set=punct_set, symbolic_root=True)
            ucorr, lcorr, total, ucm, lcm = stats
            ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
            corr_root, total_root = stats_root

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total
            dev_ucomlpete += ucm
            dev_lcomplete += lcm

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc
            dev_ucomlpete_nopunc += ucm_nopunc
            dev_lcomplete_nopunc += lcm_nopunc

            dev_root_corr += corr_root
            dev_total_root += total_root

            dev_total_inst += num_inst

        pred_writer.close()
        gold_writer.close()
        print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total,
            dev_ucomlpete * 100 / dev_total_inst, dev_lcomplete * 100 / dev_total_inst))
        print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc, dev_ucomlpete_nopunc * 100 / dev_total_inst,
            dev_lcomplete_nopunc * 100 / dev_total_inst))
        print('Root: corr: %d, total: %d, acc: %.2f%%' % (
            dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

        if dev_lcorrect_nopunc < dev_lcorr_nopunc or (
                        dev_lcorrect_nopunc == dev_lcorr_nopunc and dev_ucorrect_nopunc < dev_ucorr_nopunc):
            dev_ucorrect_nopunc = dev_ucorr_nopunc
            dev_lcorrect_nopunc = dev_lcorr_nopunc
            dev_ucomlpete_match_nopunc = dev_ucomlpete_nopunc
            dev_lcomplete_match_nopunc = dev_lcomplete_nopunc

            dev_ucorrect = dev_ucorr
            dev_lcorrect = dev_lcorr
            dev_ucomlpete_match = dev_ucomlpete
            dev_lcomplete_match = dev_lcomplete

            dev_root_correct = dev_root_corr

            best_epoch = epoch
            patient = 0
            # torch.save(network, model_name)
            torch.save(network.state_dict(), model_name)

            pred_filename = 'tmp/%spred_test%d' % (str(uid), epoch)
            pred_writer.start(pred_filename)
            gold_filename = 'tmp/%sgold_test%d' % (str(uid), epoch)
            gold_writer.start(gold_filename)

            test_ucorrect = 0.0
            test_lcorrect = 0.0
            test_ucomlpete_match = 0.0
            test_lcomplete_match = 0.0
            test_total = 0

            test_ucorrect_nopunc = 0.0
            test_lcorrect_nopunc = 0.0
            test_ucomlpete_match_nopunc = 0.0
            test_lcomplete_match_nopunc = 0.0
            test_total_nopunc = 0
            test_total_inst = 0

            test_root_correct = 0.0
            test_total_root = 0
            for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, batch_size):
                input_encoder, _ = batch
                word, char, pos, heads, types, masks, lengths = input_encoder
                heads_pred, types_pred, _, _ = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam,
                                                              leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

                word = word.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.data.cpu().numpy()
                types = types.data.cpu().numpy()

                pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

                stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types,
                                                                        word_alphabet, pos_alphabet, lengths,
                                                                        punct_set=punct_set, symbolic_root=True)
                ucorr, lcorr, total, ucm, lcm = stats
                ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
                corr_root, total_root = stats_root

                test_ucorrect += ucorr
                test_lcorrect += lcorr
                test_total += total
                test_ucomlpete_match += ucm
                test_lcomplete_match += lcm

                test_ucorrect_nopunc += ucorr_nopunc
                test_lcorrect_nopunc += lcorr_nopunc
                test_total_nopunc += total_nopunc
                test_ucomlpete_match_nopunc += ucm_nopunc
                test_lcomplete_match_nopunc += lcm_nopunc

                test_root_correct += corr_root
                test_total_root += total_root

                test_total_inst += num_inst

            pred_writer.close()
            gold_writer.close()
        else:
            if dev_ucorr_nopunc * 100 / dev_total_nopunc < dev_ucorrect_nopunc * 100 / dev_total_nopunc - 5 or patient >= schedule:
                network.load_state_dict(torch.load(model_name))
                lr = lr * decay_rate
                optim = generate_optimizer(opt, lr, network.parameters())
                patient = 0
                decay += 1
                if decay % double_schedule_decay == 0:
                    schedule *= 2
            else:
                patient += 1

        logger.info(
            '----------------------------------------------------------------------------------------------------------------------------')
        logger.info(
            'best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total, dev_lcorrect * 100 / dev_total,
                dev_ucomlpete_match * 100 / dev_total_inst, dev_lcomplete_match * 100 / dev_total_inst,
                best_epoch))
        logger.info(
            'best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                dev_ucorrect_nopunc, dev_lcorrect_nopunc, dev_total_nopunc,
                dev_ucorrect_nopunc * 100 / dev_total_nopunc, dev_lcorrect_nopunc * 100 / dev_total_nopunc,
                dev_ucomlpete_match_nopunc * 100 / dev_total_inst, dev_lcomplete_match_nopunc * 100 / dev_total_inst,
                best_epoch))
        logger.info('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            dev_root_correct, dev_total_root, dev_root_correct * 100 / dev_total_root, best_epoch))
        logger.info(
            '----------------------------------------------------------------------------------------------------------------------------')
        logger.info(
            'best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total,
                test_lcorrect * 100 / test_total,
                test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst,
                best_epoch))
        logger.info(
            'best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
                test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
                test_ucomlpete_match_nopunc * 100 / test_total_inst,
                test_lcomplete_match_nopunc * 100 / test_total_inst,
                best_epoch))
        logger.info('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
        logger.info(
            '============================================================================================================================')

        if decay == max_decay:
            break


if __name__ == '__main__':
    main()
