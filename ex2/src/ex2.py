
import sys
from math import log, pow
import itertools
from collections import defaultdict
from functools import reduce

def read_corpuses(corpuses_path, ignore):
    with open(corpuses_path, 'r') as f:
        return list(itertools.chain(
            *filter(lambda line: line!=[''],
                    [ line.strip().split(' ') for line in filter(lambda line: ignore not in line, f.readlines()) ])))

if __name__ == '__main__':
    args = sys.argv[1:]
    # read training
    development = read_corpuses(args[0], "<TRAIN")
    # read test
    test = read_corpuses(args[1], "<TEST")
    # input word
    in_w = args[2]
    # size of the language vocabulary
    v_len = 300000

    with open(args[3], 'w') as f:
        # general props
        f.write("#Output1\t{}\n".format(args[0]))
        f.write("#Output2\t{}\n".format(args[1]))
        f.write("#Output3\t{}\n".format(in_w))
        f.write("#Output4\t{}\n".format(args[3]))
        f.write("#Output5\t{}\n".format(v_len))

        def Puniform(w):
            return 1/v_len
        f.write("#Output6\t{}\n".format(Puniform(w=in_w)))
        f.write("#Output7\t{}\n".format(len(development)))

        # lidstone
        training_lid = development[:int(round(len(development)*0.9))]
        training_lid_len = len(training_lid)
        training_lid_vocabulary = set(training_lid)
        training_lid_vocabulary_len = len(training_lid_vocabulary)
        validation_lid = development[int(round(len(development)*0.9)):]
        validation_lid_len = len(validation_lid)
        validation_lid_vocabulary = set(validation_lid)
        validation_lid_vocabulary_len = len(validation_lid_vocabulary)
        test_vocabulary = set(test)
        test_vocabulary_len = len(test_vocabulary)
        f.write("#Output8\t{}\n".format(validation_lid_len))
        f.write("#Output9\t{}\n".format(training_lid_len))
        f.write("#Output10\t{}\n".format(training_lid_vocabulary_len))
        f.write("#Output11\t{}\n".format(training_lid.count(in_w)))

        def create_freq_function(training):
            training_freq_map = defaultdict(int)
            for w in training:
                training_freq_map[w] += 1
            return lambda w: training_freq_map[w] if w in training_freq_map else 0

        # freq of x in training_lid
        c = create_freq_function(training_lid)

        def test_correctness(p, training_vocabulary):
            n0 = v_len-len(training_vocabulary)
            return round(p('unseen-event')*n0 + sum([ p(w) for w in training_vocabulary ]), 2)

        # ML probability per the training set without smoothing
        def Pml(x):
            return c(x)/training_lid_len
            # return c(x)/v_len
        print("MLE sum: {}".format(test_correctness(Pml, training_lid_vocabulary)))

        f.write("#Output12\t{}\n".format(Pml(x = in_w)))
        f.write("#Output13\t{}\n".format(Pml(x = 'unseen-word')))

        # generate a certain P lidstone estimate function based on a given lambda
        def create_Plid(lambda_, r=None):
            if r is not None:
                return lambda: (r + lambda_) / (training_lid_len + (lambda_ * v_len))
            return lambda x: (c(x) + lambda_) / (training_lid_len + (lambda_ * v_len))

        # lidstone smoothed probability per the training set
        def Plid(lambda_, x):
            return create_Plid(lambda_)(x)

        f.write("#Output14\t{}\n".format(Plid(lambda_ = 0.1, x = in_w)))
        f.write("#Output15\t{}\n".format(Plid(lambda_ = 0.1, x = 'unseen-word')))

        def perplexity(p, validation_vocabulary, validation_vocabulary_len):
            return pow(2, -1*sum([ log(p(w), 2) for w in validation_vocabulary ]) / validation_vocabulary_len)
        f.write("#Output16\t{}\n".format(perplexity(
            create_Plid(lambda_ = 0.01), validation_lid, validation_lid_len)))
        f.write("#Output17\t{}\n".format(perplexity(
            create_Plid(lambda_=0.1), validation_lid, validation_lid_len)))
        f.write("#Output18\t{}\n".format(perplexity(
            create_Plid(lambda_=1), validation_lid, validation_lid_len)))

        # evaluate the optimal lambda for lidstone smoothing MLE per the training data:
        lid_optimal_lambda = -1
        min_perplexity = -1
        for lambda_ in list([x/100 for x in range(1,201)]):
            curr_perplexity = \
                perplexity(create_Plid(lambda_ = lambda_), validation_lid, validation_lid_len)
            if min_perplexity==-1 or curr_perplexity < min_perplexity:
                min_perplexity = curr_perplexity
                lid_optimal_lambda = lambda_
            print("lid sum for lambda {}: {}".format(lambda_,
                                                     test_correctness(create_Plid(lid_optimal_lambda),
                                                                      training_lid_vocabulary)))
        f.write("#Output19\t{}\n".format(lid_optimal_lambda))
        f.write("#Output20\t{}\n".format(perplexity(create_Plid(lambda_=lid_optimal_lambda),
                                                     validation_lid,
                                                     validation_lid_len)))
        print("lid sum: {}".format(test_correctness(create_Plid(lid_optimal_lambda),
                                                    training_lid_vocabulary)))

        # heldout
        training_ho = development[:round(len(development)*0.5)]
        training_ho_vocabulary = set(training_ho)
        training_ho_vocabulary_len = len(training_ho_vocabulary)
        heldout_ho = development[round(len(development)*0.5):]
        heldout_ho_vocabulary = set(heldout_ho)
        heldout_ho_vocabulary_len = len(heldout_ho_vocabulary)
        f.write("#Output21\t{}\n".format(len(training_ho)))
        f.write("#Output22\t{}\n".format(len(heldout_ho)))

        # train heldout model
        c_t = create_freq_function(training_ho)
        c_h = create_freq_function(heldout_ho)
        development_vocabulary = set(development)
        freq_to_freq_mass_in_heldout_set_map = defaultdict(int)
        def t(r):
            if r not in freq_to_freq_mass_in_heldout_set_map:
                freq_to_freq_mass_in_heldout_set_map[r] = \
                    sum([ c_h(x) if c_t(x) == r else 0 for x in development_vocabulary ])
            return freq_to_freq_mass_in_heldout_set_map[r]
        training_ho_freq_to_freq_amount_map = defaultdict(int)
        def N(r):
            if r not in training_ho_freq_to_freq_amount_map:
                training_ho_freq_to_freq_amount_map[r] = \
                    v_len-training_ho_vocabulary_len if r==0 \
                        else sum([ 1 if c_t(w) == r else 0 for w in development_vocabulary ])
            return training_ho_freq_to_freq_amount_map[r]

        # return the held out probability for a given r frequency
        def Pho_for_freq_r(r):
            return t(r) / (N(r) * len(heldout_ho))
        # return the held out probability for a word per its r frequency
        def Pho(w):
            return Pho_for_freq_r(c_t(w))
        f.write("#Output23\t{}\n".format(Pho(in_w)))
        f.write("#Output24\t{}\n".format(Pho('unseen-word')))
        print("ho sum: {}".format(test_correctness(Pho, development_vocabulary)))

        # test
        f.write("#Output25\t{}\n".format(len(test)))
        perplexity_lid_= perplexity(create_Plid(lambda_=lid_optimal_lambda), test, len(test))
        perplexity_ho_= perplexity(Pho, test, len(test))
        f.write("#Output26\t{}\n".format(perplexity_lid_))
        f.write("#Output27\t{}\n".format(perplexity_ho_))
        f.write("#Output28\t{}\n".format("L" if perplexity_lid_ < perplexity_ho_ else "H"))

        # statistics
        def f_lambda(r):
            return create_Plid(lid_optimal_lambda, r)()*training_lid_len
        def f_ho(r):
            return Pho_for_freq_r(r)*len(training_ho)
        f.write("#Output29\n")
        for r in range(10):
            f.write("{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(r, round(f_lambda(r), 5),
                                                                  round(f_ho(r), 5), round(N(r), 5), round(t(r),5)))
