#

# build the vocab/dictionary from outside to all related lexicons
# build vocab and embed jointly for multi-languages

from __future__ import print_function

import os
import sys
import argparse

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from neuronlp2.io import get_logger, conllx_stacked_data
from neuronlp2.io_multi import create_alphabets, lang_specific_word

# Usage:
# python2 examples/vocab/build_vocab.py --embed_paths [various languages' embeddings: e1 e2 ...]
# --embed_lang_ids en de es ... --data_paths [various languages' data files: ... ] --model_path <path>

#
def parse_cmd(args):
    args_parser = argparse.ArgumentParser(description='Building the alphabets/vocabularies.')
    #
    args_parser.add_argument('--embed_paths', type=str, nargs='+', help='path for word embedding dict', required=True)
    args_parser.add_argument('--embed_lang_ids', type=str, nargs='+', help='lang ids for the embeddings', required=True)
    args_parser.add_argument('--data_paths', type=str, nargs='+', help="Data files to build vocab.", required=True)
    args_parser.add_argument('--model_path', help='path for saving alphabet files.', required=True)
    res = args_parser.parse_args(args)
    return res

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
    assert len(args.embed_paths) == len(args.embed_lang_ids), "One lang id for one embed file!"
    word_embeds = [WordVectors.load(one_embed_path) for one_embed_path in args.embed_paths]
    combined_word_dict = WordVectors.combine_embeds(word_embeds, args.embed_lang_ids)
    logger.info("Final combined un-pruned embeddings size: %d." % len(combined_word_dict))
    # create vocabs
    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(args.model_path, 'alphabets/')
    assert not os.path.exists(alphabet_path), "Alphabet path exists, please build with a new path."
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_sent_length = create_alphabets(alphabet_path, args.data_paths[0], data_paths=args.data_paths[1:], embedd_dict=combined_word_dict, max_vocabulary_size=100000, creating_mode=True)
    # save filtered embed
    hit_keys = set()
    for one_w in word_alphabet.instance2index:
        if one_w in combined_word_dict:
            hit_keys.add(one_w)
        elif one_w.lower() in combined_word_dict:
            hit_keys.add(one_w.lower())
    filtered_embed = combined_word_dict.filter(hit_keys)
    filtered_embed.save(os.path.join(alphabet_path, 'joint_embed.vec'))

class WordVectors:
    def __init__(self):
        self.num_words = None
        self.embed_size = None
        self.words = []
        self.vecs = {}

    def __len__(self):
        return len(self.vecs)

    def __contains__(self, item):
        return item in self.vecs

    def has_key(self, k, lc_back=True):
        if k in self.vecs:
            return True
        elif lc_back:
            return str.lower(k) in self.vecs
        return False

    def get_vec(self, k, df=None, lc_back=True):
        if k in self.vecs:
            return self.vecs[k]
        elif lc_back:
            # back to lowercased
            lc = str.lower(k)
            if lc in self.vecs:
                return self.vecs[lc]
        return df

    def save(self, fname):
        print("Saving w2v num_words=%d, embed_size=%d to %s." % (self.num_words, self.embed_size, fname))
        with open(fname, "w") as fd:
            fd.write("%d %d\n" % (self.num_words, self.embed_size))
            for w in self.words:
                vec = self.vecs[w]
                print_list = [w.encode('utf-8')] + ["%.6f" % float(z) for z in vec]
                fd.write(" ".join(print_list)+"\n")

    def filter(self, key_set):
        one = WordVectors()
        one.num_words, one.embed_size = self.num_words, self.embed_size
        for w in self.words:
            if w in key_set:
                one.words.append(w)
                one.vecs[w] = self.vecs[w]
        one.num_words = len(one.vecs)
        print("Filter from num_words=%d/embed_size=%d to num_words=%s" % (self.num_words, self.embed_size, one.num_words))
        return one

    @staticmethod
    def load(fname):
        print("Loading pre-trained w2v from %s ..." % fname)
        one = WordVectors()
        with open(fname) as fd:
            # first line
            line = fd.readline().strip().decode('utf-8')
            try:
                one.num_words, one.embed_size = [int(x) for x in line.split()]
                print("Reading w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
                line = fd.readline().strip().decode('utf-8')
            except:
                print("Reading w2v.")
            # the rest
            while len(line) > 0:
                fields = line.split(" ")
                word, vec = fields[0], [float(x) for x in fields[1:]]
                assert word not in one.vecs, "Repeated key."
                if one.embed_size is None:
                    one.embed_size = len(vec)
                else:
                    assert len(vec) == one.embed_size, "Unmatched embed dimension."
                one.vecs[word] = vec
                one.words.append(word)
                line = fd.readline().strip().decode('utf-8')
        # final
        if one.num_words is None:
            one.num_words = len(one.vecs)
            print("Reading w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
        else:
            assert one.num_words == len(one.vecs), "Unmatched num of words."
        return one

    @staticmethod
    def combine_embeds(word_dicts, lang_ids):
        assert len(word_dicts)==len(lang_ids), "One lang id for one embed!"
        number_to_combine = len(word_dicts)
        #
        one = WordVectors()
        one.embed_size = word_dicts[0].embed_size
        print("Combining embeds of %s." % (lang_ids, ))
        for idx in range(number_to_combine):
            repeated_counts = [0 for _ in range(number_to_combine)]
            cur_embed, cur_id = word_dicts[idx], lang_ids[idx]
            for one_w in cur_embed.words:
                #
                prefixed_w = lang_specific_word(one_w, lang_id=cur_id)
                one.words.append(prefixed_w)
                one.vecs[prefixed_w] = cur_embed.vecs[one_w]
                #
                for idx2 in range(number_to_combine):
                    cur_embed2, cur_id2 = word_dicts[idx2], lang_ids[idx2]
                    if one_w in cur_embed2:
                        repeated_counts[idx2] += 1
            print("Key repeat counts of %s: %s." % (cur_id, repeated_counts))
        one.num_words = len(one.vecs)
        return one

if __name__ == '__main__':
    main()
