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

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_stacked_data, conllx_data
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser
from neuronlp2.models import StackPtrNet, BiRecurrentConvBiAffine

from neuronlp2.io_multi import guess_language_id, create_alphabets, lang_specific_word
from neuronlp2.io_multi.multi_vocab import iter_file
from neuronlp2.utils import load_embedding_dict
from neuronlp2.io.utils import DIGIT_RE

from neuronlp2.nn import Embedding

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--parser', choices=['stackptr', 'biaffine'], help='Parser', required=True)
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    args_parser.add_argument('--model_name', help='name for saving model file.', required=True)
    args_parser.add_argument('--out_filename', help='filename to save analysis results.', required=True)
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--ordered', action='store_true', help='Using order constraints in decoding')
    args_parser.add_argument('--decode', choices=['mst', 'greedy'], default='mst', help='decoding algorithm')
    args_parser.add_argument('--display', action='store_true', help='Display wrong examples')
    args_parser.add_argument('--gpu', action='store_true', help='Using GPU')
    #
    args_parser.add_argument('--extra_embed', type=str, help="Path for extra embedding file for extra language testing.")
    args_parser.add_argument('--extra_embed_src', type=str, help="Path for extra embedding file for src language (maybe need adding new ones).")

    args = args_parser.parse_args()

    logger = get_logger("Analyzer")

    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name

    punct_set = None
    punctuation = args.punctuation
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    use_gpu = args.gpu
    parser = args.parser

    if parser == 'stackptr':
        stackptr(model_path, model_name, test_path, punct_set, use_gpu, logger, args)
    elif parser == 'biaffine':
        biaffine(model_path, model_name, test_path, punct_set, use_gpu, logger, args)
    else:
        raise ValueError('Unknown parser: %s' % parser)

# ==========
# about augmenting the vocabs and embeddings with the current test file and extra embeddings
def augment_with_extra_embedding(the_alphabet, extra_embed_file, extra_embed_src_file, test_file, logger):
    extra_embeds_arr = []
    if extra_embed_file is not None:
        # reopen the vocab
        the_alphabet.open()
        # read the embed
        extra_word_dict, _ = load_embedding_dict('word2vec', extra_embed_file)
        if extra_embed_src_file is not None:
            src_extra_word_dict, _ = load_embedding_dict('word2vec', extra_embed_src_file)
        lang_id = guess_language_id(test_file)
        for one_sent in iter_file(test_file):
            for w in one_sent["word"]:
                already_spec = w.startswith("!en_")
                if already_spec:
                    normed_word = w
                else:
                    normed_word = DIGIT_RE.sub(b"0", w)
                    normed_word = lang_specific_word(normed_word, lang_id=lang_id)
                #
                if normed_word in the_alphabet.instance2index:
                    continue
                # TODO: assume english is the source for run-translate
                if already_spec:
                    w = w[4:]
                    check_dict = src_extra_word_dict
                else:
                    check_dict = extra_word_dict
                #
                if w in check_dict:
                    new_embed_arr = check_dict[w]
                elif w.lower() in check_dict:
                    new_embed_arr = check_dict[w.lower()]
                else:
                    new_embed_arr = None
                if new_embed_arr is not None:
                    extra_embeds_arr.append(new_embed_arr)
                    the_alphabet.add(normed_word)
        # close the vocab
        the_alphabet.close()
    logger.info("Augmenting the vocab with new words of %s, now vocab is %s." % (len(extra_embeds_arr), the_alphabet.size()))
    return extra_embeds_arr

def augment_network_embed(new_size, network, extra_embeds_arr):
    if len(extra_embeds_arr) == 0:
        return
    old_embed = network.word_embedd
    if old_embed is None:
        return
    old_arr = old_embed.weight.data.cpu().numpy()
    new_arr = np.concatenate([old_arr, extra_embeds_arr], 0)
    assert new_arr.shape[0] == new_size
    new_embed = Embedding(new_size, new_arr.shape[1], init_embedding=torch.from_numpy(new_arr))
    network.word_embedd = new_embed

# ==========

def biaffine(model_path, model_name, test_path, punct_set, use_gpu, logger, args):
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_data.create_alphabets(alphabet_path,
        None, data_paths=[None, None], max_vocabulary_size=50000, embedd_dict=None)
    # word_alphabet, char_alphabet, pos_alphabet, type_alphabet = create_alphabets(alphabet_path,
    #     None, data_paths=[None, None], max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    decoding = args.decode
    out_filename = args.out_filename

    logger.info('use gpu: %s, decoding: %s' % (use_gpu, decoding))

    #
    extra_embeds_arr = augment_with_extra_embedding(word_alphabet, args.extra_embed, args.extra_embed_src, test_path, logger)

    # ===== the reading
    def _read_one(path, is_train):
        lang_id = guess_language_id(path)
        logger.info("Reading: guess that the language of file %s is %s." % (path, lang_id))
        one_data = conllx_data.read_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, use_gpu=use_gpu, volatile=(not is_train), symbolic_root=True, lang_id=lang_id)
        return one_data

    data_test = _read_one(test_path, False)

    # data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
    #                                               use_gpu=use_gpu, volatile=True, symbolic_root=True)

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    logger.info('model: %s' % model_name)

    def load_model_arguments_from_json():
        arguments = json.load(open(arg_path, 'r'))
        return arguments['args'], arguments['kwargs']

    arg_path = model_name + '.arg.json'
    args, kwargs = load_model_arguments_from_json()
    network = BiRecurrentConvBiAffine(use_gpu=use_gpu, *args, **kwargs)
    network.load_state_dict(torch.load(model_name))

    #
    augment_network_embed(word_alphabet.size(), network, extra_embeds_arr)

    if use_gpu:
        network.cuda()
    else:
        network.cpu()

    network.eval()

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

    if decoding == 'greedy':
        decode = network.decode
    elif decoding == 'mst':
        decode = network.decode_mst
    else:
        raise ValueError('Unknown decoding algorithm: %s' % decoding)

    # pred_writer.start('tmp/analyze_pred_%s' % str(uid))
    # gold_writer.start('tmp/analyze_gold_%s' % str(uid))
    # pred_writer.start(model_path + out_filename + '_pred')
    # gold_writer.start(model_path + out_filename + '_gold')
    pred_writer.start(out_filename + '_pred')
    gold_writer.start(out_filename + '_gold')

    sent = 0
    start_time = time.time()

    for batch in conllx_data.iterate_batch_variable(data_test, 1):
        sys.stdout.write('%d, ' % sent)
        sys.stdout.flush()
        sent += 1

        word, char, pos, heads, types, masks, lengths = batch
        heads_pred, types_pred = decode(word, char, pos, mask=masks, length=lengths,
                                        leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
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

    print('\ntime: %.2fs' % (time.time() - start_time))
    print('test W. Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
        test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst))
    print('test Wo Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
        test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
        test_ucomlpete_match_nopunc * 100 / test_total_inst, test_lcomplete_match_nopunc * 100 / test_total_inst))
    print('test Root: corr: %d, total: %d, acc: %.2f%%' % (
        test_root_correct, test_total_root, test_root_correct * 100 / test_total_root))


def stackptr(model_path, model_name, test_path, punct_set, use_gpu, logger, args):
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, \
    max_sent_length = conllx_stacked_data.create_alphabets(alphabet_path, None, data_paths=[None, None],
                                                           max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    beam = args.beam
    ordered = args.ordered
    display_inst = args.display
    out_filename = args.out_filename
    extra_embed = args.extra_embed

    def load_model_arguments_from_json():
        arguments = json.load(open(arg_path, 'r'))
        return arguments['args'], arguments['kwargs']

    arg_path = model_name + '.arg.json'
    args, kwargs = load_model_arguments_from_json()

    prior_order = kwargs['prior_order']
    logger.info('use gpu: %s, beam: %d, order: %s (%s)' % (use_gpu, beam, prior_order, ordered))

    #
    extra_embeds_arr = augment_with_extra_embedding(word_alphabet, extra_embed, args.extra_embed_src, test_path, logger)

    # ===== the reading
    def _read_one(path, is_train):
        lang_id = guess_language_id(path)
        logger.info("Reading: guess that the language of file %s is %s." % (path, lang_id))
        one_data = conllx_stacked_data.read_stacked_data_to_variable(path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, use_gpu=use_gpu, volatile=(not is_train), prior_order=prior_order, lang_id=lang_id)
        return one_data

    data_test = _read_one(test_path, False)

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    logger.info('model: %s' % model_name)
    network = StackPtrNet(use_gpu=use_gpu, *args, **kwargs)
    network.load_state_dict(torch.load(model_name))

    #
    augment_network_embed(word_alphabet.size(), network, extra_embeds_arr)

    if use_gpu:
        network.cuda()
    else:
        network.cpu()

    network.eval()

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

    test_ucorrect_stack_leaf = 0.0
    test_ucorrect_stack_non_leaf = 0.0

    test_lcorrect_stack_leaf = 0.0
    test_lcorrect_stack_non_leaf = 0.0

    test_leaf = 0
    test_non_leaf = 0

    # pred_writer.start('tmp/analyze_pred_%s' % str(uid))
    # gold_writer.start('tmp/analyze_gold_%s' % str(uid))
    # pred_writer.start(model_path + out_filename + '_pred')
    # gold_writer.start(model_path + out_filename + '_gold')
    pred_writer.start(out_filename + '_pred')
    gold_writer.start(out_filename + '_gold')

    sent = 0
    start_time = time.time()
    for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, 1):
        sys.stdout.write('%d, ' % sent)
        sys.stdout.flush()
        sent += 1

        input_encoder, input_decoder = batch
        word, char, pos, heads, types, masks, lengths = input_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, mask_d, lengths_d = input_decoder
        heads_pred, types_pred, children_pred, stacked_types_pred = network.decode(word, char, pos, mask=masks,
                                                                                   length=lengths, beam=beam,
                                                                                   ordered=ordered,
                                                                                   leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

        stacked_heads = stacked_heads.data
        children = children.data
        stacked_types = stacked_types.data
        children_pred = torch.from_numpy(children_pred).long()
        stacked_types_pred = torch.from_numpy(stacked_types_pred).long()
        if use_gpu:
            children_pred = children_pred.cuda()
            stacked_types_pred = stacked_types_pred.cuda()
        mask_d = mask_d.data
        mask_leaf = torch.eq(children, stacked_heads).float()
        mask_non_leaf = (1.0 - mask_leaf)
        mask_leaf = mask_leaf * mask_d
        mask_non_leaf = mask_non_leaf * mask_d
        num_leaf = mask_leaf.sum()
        num_non_leaf = mask_non_leaf.sum()

        ucorr_stack = torch.eq(children_pred, children).float()
        lcorr_stack = ucorr_stack * torch.eq(stacked_types_pred, stacked_types).float()
        ucorr_stack_leaf = (ucorr_stack * mask_leaf).sum()
        ucorr_stack_non_leaf = (ucorr_stack * mask_non_leaf).sum()

        lcorr_stack_leaf = (lcorr_stack * mask_leaf).sum()
        lcorr_stack_non_leaf = (lcorr_stack * mask_non_leaf).sum()

        test_ucorrect_stack_leaf += ucorr_stack_leaf
        test_ucorrect_stack_non_leaf += ucorr_stack_non_leaf
        test_lcorrect_stack_leaf += lcorr_stack_leaf
        test_lcorrect_stack_non_leaf += lcorr_stack_non_leaf

        test_leaf += num_leaf
        test_non_leaf += num_non_leaf

        # ------------------------------------------------------------------------------------------------

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

    print('\ntime: %.2fs' % (time.time() - start_time))

    print('test W. Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total, test_lcorrect * 100 / test_total,
        test_ucomlpete_match * 100 / test_total_inst, test_lcomplete_match * 100 / test_total_inst))
    print('test Wo Punct:  ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
        test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
        test_ucomlpete_match_nopunc * 100 / test_total_inst, test_lcomplete_match_nopunc * 100 / test_total_inst))
    print('test Root: corr: %d, total: %d, acc: %.2f%%' % (
        test_root_correct, test_total_root, test_root_correct * 100 / test_total_root))
    print(
        '============================================================================================================================')

    print('Stack leaf:     ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
        test_ucorrect_stack_leaf, test_lcorrect_stack_leaf, test_leaf,
        test_ucorrect_stack_leaf * 100 / test_leaf, test_lcorrect_stack_leaf * 100 / test_leaf))
    print('Stack non_leaf: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
        test_ucorrect_stack_non_leaf, test_lcorrect_stack_non_leaf, test_non_leaf,
        test_ucorrect_stack_non_leaf * 100 / test_non_leaf, test_lcorrect_stack_non_leaf * 100 / test_non_leaf))
    print(
        '============================================================================================================================')

    def analyze():
        np.set_printoptions(linewidth=100000)
        # pred_path = 'tmp/analyze_pred_%s' % str(uid)
        pred_path = model_path + out_filename + '_pred'
        data_gold = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet,
                                                                      pos_alphabet, type_alphabet,
                                                                      use_gpu=use_gpu, volatile=True,
                                                                      prior_order=prior_order)
        data_pred = conllx_stacked_data.read_stacked_data_to_variable(pred_path, word_alphabet, char_alphabet,
                                                                      pos_alphabet, type_alphabet,
                                                                      use_gpu=use_gpu, volatile=True,
                                                                      prior_order=prior_order)

        gold_iter = conllx_stacked_data.iterate_batch_stacked_variable(data_gold, 1)
        test_iter = conllx_stacked_data.iterate_batch_stacked_variable(data_pred, 1)
        model_err = 0
        search_err = 0
        type_err = 0
        for gold, pred in zip(gold_iter, test_iter):
            gold_encoder, gold_decoder = gold
            word, char, pos, gold_heads, gold_types, masks, lengths = gold_encoder
            gold_stacked_heads, gold_children, gold_siblings, gold_stacked_types, gold_skip_connect, gold_mask_d, gold_lengths_d = gold_decoder

            pred_encoder, pred_decoder = pred
            _, _, _, pred_heads, pred_types, _, _ = pred_encoder
            pred_stacked_heads, pred_children, pred_siblings, pred_stacked_types, pred_skip_connect, pred_mask_d, pred_lengths_d = pred_decoder

            assert gold_heads.size() == pred_heads.size(), 'sentence dis-match.'

            ucorr_stack = torch.eq(pred_children, gold_children).float()
            lcorr_stack = ucorr_stack * torch.eq(pred_stacked_types, gold_stacked_types).float()
            ucorr_stack = (ucorr_stack * gold_mask_d).data.sum()
            lcorr_stack = (lcorr_stack * gold_mask_d).data.sum()
            num_stack = gold_mask_d.data.sum()

            if lcorr_stack < num_stack:
                loss_pred, loss_pred_arc, loss_pred_type = calc_loss(network, word, char, pos, pred_heads,
                                                                     pred_stacked_heads, pred_children, pred_siblings,
                                                                     pred_stacked_types,
                                                                     pred_skip_connect, masks, lengths, pred_mask_d,
                                                                     pred_lengths_d)

                loss_gold, loss_gold_arc, loss_gold_type = calc_loss(network, word, char, pos, gold_heads,
                                                                     gold_stacked_heads, gold_children, gold_siblings,
                                                                     gold_stacked_types,
                                                                     gold_skip_connect, masks, lengths, gold_mask_d,
                                                                     gold_lengths_d)

                if display_inst:
                    print('%d, %d, %d' % (ucorr_stack, lcorr_stack, num_stack))
                    print('pred(arc, type): %.4f (%.4f, %.4f), gold(arc, type): %.4f (%.4f, %.4f)' % (
                        loss_pred, loss_pred_arc, loss_pred_type, loss_gold, loss_gold_arc, loss_gold_type))
                    word = word[0].data.cpu().numpy()
                    pos = pos[0].data.cpu().numpy()
                    head_gold = gold_heads[0].data.cpu().numpy()
                    type_gold = gold_types[0].data.cpu().numpy()
                    head_pred = pred_heads[0].data.cpu().numpy()
                    type_pred = pred_types[0].data.cpu().numpy()
                    display(word, pos, head_gold, type_gold, head_pred, type_pred, lengths[0], word_alphabet,
                            pos_alphabet, type_alphabet)

                    length_dec = gold_lengths_d[0]
                    gold_display = np.empty([3, length_dec])
                    gold_display[0] = gold_stacked_types.data[0].cpu().numpy()[:length_dec]
                    gold_display[1] = gold_children.data[0].cpu().numpy()[:length_dec]
                    gold_display[2] = gold_stacked_heads.data[0].cpu().numpy()[:length_dec]
                    print(gold_display)
                    print('--------------------------------------------------------')
                    pred_display = np.empty([3, pred_lengths_d[0]])[:length_dec]
                    pred_display[0] = pred_stacked_types.data[0].cpu().numpy()[:length_dec]
                    pred_display[1] = pred_children.data[0].cpu().numpy()[:length_dec]
                    pred_display[2] = pred_stacked_heads.data[0].cpu().numpy()[:length_dec]
                    print(pred_display)
                    print('========================================================')
                    raw_input()

                if ucorr_stack == num_stack:
                    type_err += 1
                elif loss_pred < loss_gold:
                    model_err += 1
                else:
                    search_err += 1
        print('type   errors: %d' % type_err)
        print('model  errors: %d' % model_err)
        print('search errors: %d' % search_err)

    analyze()


def calc_loss(network, word, char, pos, heads, stacked_heads, children, sibling, stacked_types, skip_connect, mask_e,
              length_e, mask_d, length_d):
    loss_arc_leaf, loss_arc_non_leaf, \
    loss_type_leaf, loss_type_non_leaf, \
    loss_cov, num_leaf, num_non_leaf = network.loss(word, char, pos, heads, stacked_heads, children, sibling,
                                                    stacked_types, 1.0, skip_connect=skip_connect,
                                                    mask_e=mask_e, length_e=length_e, mask_d=mask_d, length_d=length_d)

    num_leaf = num_leaf.data[0]
    num_non_leaf = num_non_leaf.data[0]

    err_arc_leaf = loss_arc_leaf.data[0] * num_leaf
    err_arc_non_leaf = loss_arc_non_leaf.data[0] * num_non_leaf

    err_type_leaf = loss_type_leaf.data[0] * num_leaf
    err_type_non_leaf = loss_type_non_leaf.data[0] * num_non_leaf

    err_cov = loss_cov.data[0] * (num_leaf + num_non_leaf)

    err_arc = err_arc_leaf + err_arc_non_leaf
    err_type = err_type_leaf + err_type_non_leaf

    err = err_arc + err_type

    return err, err_arc, err_type


def display(word, pos, head_gold, type_gold, head_pred, type_pred, length, word_alphabet, pos_alphabet, type_alphabet):
    for j in range(0, length):
        w = word_alphabet.get_instance(word[j]).encode('utf-8')
        p = pos_alphabet.get_instance(pos[j]).encode('utf-8')
        t_g = type_alphabet.get_instance(type_gold[j]).encode('utf-8')
        h_g = head_gold[j]
        t_p = type_alphabet.get_instance(type_pred[j]).encode('utf-8')
        h_p = head_pred[j]
        print('%d\t%s\t%s\t%d\t%s\t%d\t%s\n' % (j, w, p, h_g, t_g, h_p, t_p))
    print('-----------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
