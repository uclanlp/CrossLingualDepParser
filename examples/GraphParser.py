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
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvBiAffine
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser

from neuronlp2.io_multi import guess_language_id

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducibility')
    args_parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU', 'FastLSTM'],
                             help='architecture of rnn', required=True)
    args_parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_parser.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--type_space', type=int, default=128, help='Dimension of tag space')
    args_parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of encoder.')
    args_parser.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_parser.add_argument('--pos', action='store_true', help='use part-of-speech embedding.')
    args_parser.add_argument('--char', action='store_true', help='use character embedding and CNN.')
    args_parser.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_parser.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm')
    args_parser.add_argument('--objective', choices=['cross_entropy', 'crf'], default='cross_entropy', help='objective function of training procedure.')
    args_parser.add_argument('--decode', choices=['mst', 'greedy'], default='mst', help='decoding algorithm')
    args_parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    # args_parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--p_rnn', nargs='+', type=float, required=True, help='dropout rate for RNN')
    args_parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    # args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--word_embedding', choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters', required=True)
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    args_parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--vocab_path', help='path for prebuilt alphabets.', default=None)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    #
    args_parser.add_argument('--no_word', action='store_true', help='do not use word embedding.')
    #
    # lrate schedule with warmup in the first iter.
    args_parser.add_argument('--use_warmup_schedule', action='store_true', help="Use warmup lrate schedule.")
    args_parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate of learning rate')
    args_parser.add_argument('--max_decay', type=int, default=9, help='Number of decays before stop')
    args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--double_schedule_decay', type=int, default=5, help='Number of decays to double schedule')
    args_parser.add_argument('--check_dev', type=int, default=5,
                             help='Check development performance in every n\'th iteration')
    # Tansformer encoder
    args_parser.add_argument('--no_CoRNN', action='store_true', help='do not use context RNN.')
    args_parser.add_argument('--trans_hid_size', type=int, default=1024,
                             help='#hidden units in point-wise feed-forward in transformer')
    args_parser.add_argument('--d_k', type=int, default=64, help='d_k for multi-head-attention in transformer encoder')
    args_parser.add_argument('--d_v', type=int, default=64, help='d_v for multi-head-attention in transformer encoder')
    args_parser.add_argument('--multi_head_attn', action='store_true', help='use multi-head-attention.')
    args_parser.add_argument('--num_head', type=int, default=8, help='Value of h in multi-head attention')
    # - positional
    args_parser.add_argument('--enc_use_neg_dist', action='store_true',
                             help="Use negative distance for enc's relational-distance embedding.")
    args_parser.add_argument('--enc_clip_dist', type=int, default=0,
                             help="The clipping distance for relative position features.")
    args_parser.add_argument('--position_dim', type=int, default=50, help='Dimension of Position embeddings.')
    args_parser.add_argument('--position_embed_num', type=int, default=200,
                             help='Minimum value of position embedding num, which usually is max-sent-length.')
    args_parser.add_argument('--train_position', action='store_true', help='train positional encoding for transformer.')
    #
    args_parser.add_argument('--train_len_thresh', type=int, default=100, help='In training, discard sentences longer than this.')

    #
    args = args_parser.parse_args()

    # fix data-prepare seed
    random.seed(1234)
    np.random.seed(1234)
    # model's seed
    torch.manual_seed(args.seed)

    logger = get_logger("GraphParser")

    mode = args.mode
    obj = args.objective
    decoding = args.decode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    arc_space = args.arc_space
    type_space = args.type_space
    num_layers = args.num_layers
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    opt = args.opt
    momentum = 0.9
    betas = (0.9, 0.9)
    eps = args.epsilon
    decay_rate = args.decay_rate
    clip = args.clip
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    punctuation = args.punctuation

    freeze = args.freeze
    word_embedding = args.word_embedding
    word_path = args.word_path

    use_char = args.char
    char_embedding = args.char_embedding
    char_path = args.char_path

    use_pos = args.pos
    pos_dim = args.pos_dim
    word_dict, word_dim = utils.load_embedding_dict(word_embedding, word_path)
    char_dict = None
    char_dim = args.char_dim
    if char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(char_embedding, char_path)

    #
    vocab_path = args.vocab_path if args.vocab_path is not None else args.model_path

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(vocab_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    # todo(warn): exactly same for loading vocabs
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_data.create_alphabets(alphabet_path, train_path, data_paths=[dev_path, test_path], max_vocabulary_size=50000, embedd_dict=word_dict)

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
        one_data = conllx_data.read_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                    use_gpu=use_gpu, volatile=(not is_train), symbolic_root=True, lang_id=lang_id,
                                                     len_thresh=(args.train_len_thresh if is_train else 100000))
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
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                embedding = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    window = 3
    if obj == 'cross_entropy':
        network = BiRecurrentConvBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                                          mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                          embedd_word=word_table, embedd_char=char_table,
                                          p_in=p_in, p_out=p_out, p_rnn=p_rnn, biaffine=True, pos=use_pos, char=use_char,
                                          train_position=args.train_position, use_con_rnn=(not args.no_CoRNN), trans_hid_size=args.trans_hid_size,
                                          d_k=args.d_k, d_v=args.d_v, multi_head_attn=args.multi_head_attn, num_head=args.num_head,
                                          enc_use_neg_dist=args.enc_use_neg_dist, enc_clip_dist=args.enc_clip_dist,
                                          position_dim=args.position_dim, max_sent_length=max_sent_length,
                                          use_gpu=use_gpu, no_word=args.no_word)

    elif obj == 'crf':
        raise NotImplementedError
    else:
        raise RuntimeError('Unknown objective: %s' % obj)

    def save_args():
        arg_path = model_name + '.arg.json'
        arguments = [word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, window,
                     mode, hidden_size, num_layers, num_types, arc_space, type_space]
        kwargs = {'p_in': p_in, 'p_out': p_out, 'p_rnn': p_rnn, 'biaffine': True, 'pos': use_pos, 'char': use_char,
                  'train_position': args.train_position, 'use_con_rnn': (not args.no_CoRNN), 'trans_hid_size': args.trans_hid_size,
                  'd_k': args.d_k, 'd_v': args.d_v, 'multi_head_attn': args.multi_head_attn, 'num_head': args.num_head,
                  'enc_use_neg_dist': args.enc_use_neg_dist, 'enc_clip_dist': args.enc_clip_dist,
                  'position_dim': args.position_dim, 'max_sent_length': max_sent_length, 'no_word': args.no_word}
        json.dump({'args': arguments, 'kwargs': kwargs}, open(arg_path, 'w'), indent=4)

    if freeze:
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
    logger.info("Embedding dim: word=%d (%s), char=%d (%s), pos=%d (%s)" % (word_dim, word_status, char_dim, char_status, pos_dim, pos_status))
    logger.info("CNN: filter=%d, kernel=%d" % (num_filters, window))
    logger.info("RNN: %s, num_layer=%d, hidden=%d, arc_space=%d, type_space=%d" % (mode, num_layers, hidden_size, arc_space, type_space))
    logger.info("train: obj: %s, l2: %f, (#data: %d, batch: %d, clip: %.2f, unk replace: %.2f)" % (obj, gamma, num_data, batch_size, clip, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    logger.info("decoding algorithm: %s" % decoding)
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

    if decoding == 'greedy':
        decode = network.decode
    elif decoding == 'mst':
        decode = network.decode_mst
    else:
        raise ValueError('Unknown decoding algorithm: %s' % decoding)

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
        print('Epoch %d (%s, optim: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, patient=%d, decay=%d)): ' % (epoch, mode, opt, lr, eps, decay_rate, schedule, patient, decay))
        train_err = 0.
        train_err_arc = 0.
        train_err_type = 0.
        train_total = 0.
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
            #
            word, char, pos, heads, types, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size, unk_replace=unk_replace)

            optim.zero_grad()
            loss_arc, loss_type = network.loss(word, char, pos, heads, types, mask=masks, length=lengths)
            loss = loss_arc + loss_type
            loss.backward()
            clip_grad_norm(network.parameters(), clip)
            optim.step()

            num_inst = word.size(0) if obj == 'crf' else masks.data.sum() - word.size(0)
            train_err += loss.data[0] * num_inst
            train_err_arc += loss_arc.data[0] * num_inst
            train_err_type += loss_type.data[0] * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 10 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, arc: %.4f, type: %.4f, time left: %.2fs' % (batch, num_batches, train_err / train_total,
                                                                                                 train_err_arc / train_total, train_err_type / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, arc: %.4f, type: %.4f, time: %.2fs' % (num_batches, train_err / train_total, train_err_arc / train_total, train_err_type / train_total, time.time() - start_time))

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
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, pos, heads, types, masks, lengths = batch
            heads_pred, types_pred = decode(word, char, pos, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            word = word.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
            lengths = lengths.cpu().numpy()
            heads = heads.data.cpu().numpy()
            types = types.data.cpu().numpy()

            pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
            gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

            stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types, word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
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
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total, dev_ucomlpete * 100 / dev_total_inst, dev_lcomplete * 100 / dev_total_inst))
        print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc,
            dev_ucomlpete_nopunc * 100 / dev_total_inst, dev_lcomplete_nopunc * 100 / dev_total_inst))
        print('Root: corr: %d, total: %d, acc: %.2f%%' %(dev_root_corr, dev_total_root, dev_root_corr * 100 / dev_total_root))

        if dev_lcorrect_nopunc < dev_lcorr_nopunc or (dev_lcorrect_nopunc == dev_lcorr_nopunc and dev_ucorrect_nopunc < dev_ucorr_nopunc):
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
            for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                word, char, pos, heads, types, masks, lengths = batch
                heads_pred, types_pred = decode(word, char, pos, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                word = word.data.cpu().numpy()
                pos = pos.data.cpu().numpy()
                lengths = lengths.cpu().numpy()
                heads = heads.data.cpu().numpy()
                types = types.data.cpu().numpy()

                pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
                gold_writer.write(word, pos, heads, types, lengths, symbolic_root=True)

                stats, stats_nopunc, stats_root, num_inst = parser.eval(word, pos, heads_pred, types_pred, heads, types, word_alphabet, pos_alphabet, lengths, punct_set=punct_set, symbolic_root=True)
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
                # network = torch.load(model_name)
                network.load_state_dict(torch.load(model_name))
                lr = lr * decay_rate
                optim = generate_optimizer(opt, lr, network.parameters())

                if decoding == 'greedy':
                    decode = network.decode
                elif decoding == 'mst':
                    decode = network.decode_mst
                else:
                    raise ValueError('Unknown decoding algorithm: %s' % decoding)

                patient = 0
                decay += 1
                if decay % double_schedule_decay == 0:
                    schedule *= 2
            else:
                patient += 1

        print('----------------------------------------------------------------------------------------------------------------------------')
        print('best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total, dev_lcorrect * 100 / dev_total,
            dev_ucomlpete_match * 100 / dev_total_inst, dev_lcomplete_match * 100 / dev_total_inst,
            best_epoch))
        print('best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            dev_ucorrect_nopunc, dev_lcorrect_nopunc, dev_total_nopunc,
            dev_ucorrect_nopunc * 100 / dev_total_nopunc, dev_lcorrect_nopunc * 100 / dev_total_nopunc,
            dev_ucomlpete_match_nopunc * 100 / dev_total_inst, dev_lcomplete_match_nopunc * 100 / dev_total_inst,
            best_epoch))
        print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            dev_root_correct, dev_total_root, dev_root_correct * 100 / dev_total_root, best_epoch))
        print('----------------------------------------------------------------------------------------------------------------------------')
        print('best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
            test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst,
            best_epoch))
        print('best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
            test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
            test_ucomlpete_match_nopunc * 100 / test_total_inst, test_lcomplete_match_nopunc * 100 / test_total_inst,
            best_epoch))
        print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
        print('============================================================================================================================')

        if decay == max_decay:
            break


if __name__ == '__main__':
    main()
