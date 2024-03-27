import utils.glob as glob
logger = glob.get('logger')

import numpy as np
from collections import Counter

from tqdm import tqdm

from tools.prob import get_estimator

class NGramModel:
    def __init__(self, train_data, hp, setting) -> None:
        self.train_data = train_data
        self.hp = hp
        self.setting = setting

        self.N = hp['N']

        self.ngram_cnts = Counter()
        self.context_cnts = Counter()
        self.word_cnts = Counter()
        self.vocab_size = 0
        self.train()

        estimator_hp = {
            'smooth': hp['smooth'],
            'ngram_cnts': self.ngram_cnts,
            'context_cnts': self.context_cnts,
            'vocab_size': self.vocab_size
        }
        self.estimator = get_estimator(estimator_hp)

    def train(self):
        sentences = self.train_data
        loop = tqdm(sentences, desc='[Training]')
        for sentence in loop:
            for i in range(len(sentence) - self.N + 1):
                ngram = tuple(sentence[i:i + self.N])
                context = ngram[:-1]
                word = ngram[-1]
                self.ngram_cnts[ngram] += 1
                self.context_cnts[context] += 1
                self.word_cnts[word] += 1
        self.vocab_size = len(self.word_cnts)

    def perplexity(self, test_data):
        perplexity = []
        for sentence in test_data:
            sentence_prob = []
            for i in range(len(sentence) - self.N + 1):
                ngram = tuple(sentence[i:i + self.N])
                sentence_prob.append(self.estimator(ngram))
            sentence_prob = np.array(sentence_prob)
            product = np.prod(sentence_prob)
            pp = product ** (-1 / len(sentence))
            perplexity = np.append(perplexity, pp)

        return perplexity

