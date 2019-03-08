#

# build the vocab/dictionary from outside to all related lexicons

from __future__ import print_function

import os
import sys
import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from neuronlp2 import utils
from neuronlp2.io import get_logger, conllx_stacked_data


#
# Only use for building multi-lingual vocabs, this is only a simple workaround.
# However, we might also want multi-lingual embeddings before training for convenience.
# Usage:
# python2 examples/vocab/build_vocab.py --word_embedding <embde-type> --word_paths [various languages' embeddings: e1 e2 ...]
# --train <english-train-file> --extra [various languages' test-files: ... ] --model_path <path>

#
def parse_cmd(args):
    args_parser = argparse.ArgumentParser(description='Building the alphabets/vocabularies.')
    #
    args_parser.add_argument('--word_embedding', type=str, choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words', required=True)
    args_parser.add_argument('--word_paths', type=str, nargs='+', help='path for word embedding dict', required=True)
    args_parser.add_argument('--train', type=str, help="The main file to build vocab.", required=True)
    args_parser.add_argument('--extra', type=str, nargs='+', help="Extra files to build vocab, usually dev/tests.",
                             required=True)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    res = args_parser.parse_args(args)
    return res


def _get_keys(wd):
    try:
        return wd.keys()
    except:
        # Word2VecKeyedVectors
        return wd.vocab.keys()


# todo(warn): if not care about the specific language of the embeddings
def combine_embeds(word_dicts):
    num_dicts = len(word_dicts)
    count_ins, count_repeats = [0 for _ in range(num_dicts)], [0 for _ in range(num_dicts)]
    res = dict()
    for idx, one in enumerate(word_dicts):
        for k in _get_keys(one):
            if k in res:
                count_repeats[idx] += 1
            else:
                count_ins[idx] += 1
                res[k] = 0
    return res, count_ins, count_repeats


def main(a=None):
    if a is None:
        a = sys.argv[1:]
    args = parse_cmd(a)
    # if output directory doesn't exist, create it
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    logger = get_logger("VocabBuilder", args.model_path + '/vocab.log.txt')
    logger.info('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
    logger.info('{0}\n'.format(args))
    # load embeds
    logger.info("Load embeddings")
    word_dicts = []
    word_dim = None
    for one in args.word_paths:
        one_word_dict, one_word_dim = utils.load_embedding_dict(args.word_embedding, one)
        assert word_dim is None or word_dim == one_word_dim, "Embedding size not matched!"
        word_dicts.append(one_word_dict)
        word_dim = one_word_dim
    # combine embeds
    combined_word_dict, count_ins, count_repeats = combine_embeds(word_dicts)
    logger.info("Final embeddings size: %d." % len(combined_word_dict))
    for one_fname, one_count_ins, one_count_repeats in zip(args.word_paths, count_ins, count_repeats):
        logger.info(
            "For embed-file %s, count-in: %d, repeat-discard: %d." % (one_fname, one_count_ins, one_count_repeats))
    # create vocabs
    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(args.model_path, 'alphabets/')
    assert not os.path.exists(alphabet_path), "Alphabet path exists, please build with a new path."
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = conllx_stacked_data.create_alphabets(
        alphabet_path, args.train, data_paths=args.extra, max_vocabulary_size=100000, embedd_dict=combined_word_dict)
    # printing info
    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()
    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)


if __name__ == '__main__':
    main()
