import copy
from collections import Counter


def get_estimator(hp):
    if hp['smooth'] == 'none':
        return no_smooth(hp['ngram_cnts'], hp['context_cnts'])
    elif hp['smooth'] == 'add1':
        return add1_smooth(hp['ngram_cnts'], hp['context_cnts'], hp['vocab_size'])
    elif hp['smooth'] == 'gt':
        return gt_smooth(hp['ngram_cnts'], hp['context_cnts'])
    else:
        raise ValueError(f"Unknown smoothing method: {hp['smooth']}")


class ProbEstimator:
    def __init__(self, ngram_cnts, context_cnts):
        self.ngram_cnts = ngram_cnts
        self.context_cnts = context_cnts

# ----------------------- no smooth ---------------------- #
class no_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts):
        super().__init__(ngram_cnts, context_cnts)

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram]
        divisor = self.context_cnts[context]
        prob = dividend / divisor
        return prob

# ---------------------- add1 smooth --------------------- #
class add1_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts, vocab_size):
        super().__init__(ngram_cnts, context_cnts)
        self.vocab_size = vocab_size

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram] + 1
        divisor = self.context_cnts[context] + self.vocab_size
        prob = dividend / divisor
        return prob

# ----------------------- gt smooth ---------------------- #
class gt_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts, threshold=9):
        super().__init__(ngram_cnts, context_cnts)
        self.ngram_cnts = self.get_new_cnts(ngram_cnts, threshold)
        self.context_cnts = self.get_new_cnts(context_cnts, threshold)

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram] if self.ngram_cnts[ngram] else self.ngram_cnts['<Nr0>']
        divisor = self.context_cnts[context] if self.context_cnts[context] else self.context_cnts['<Nr0>']
        prob = dividend / divisor
        return prob

    def get_new_cnts(self, cnts, threshold):
        # get dr
        Nr = Counter(cnts.values())
        dr = Counter()
        k = threshold
        for r in range(1, k+1):
            dr[r] = (r+1) * Nr[r+1] / Nr[r]
            dr[r] -= r * (k+1) * Nr[k+1] / Nr[1]
            dr[r] /= 1 - (k+1) * Nr[k+1] / Nr[1]
        # get new cnts
        new_cnts = copy.deepcopy(cnts)
        for key, value in new_cnts.items():
            if value <= k:
                new_cnts[key] = dr[value]
        new_cnts['<Nr0>'] = Nr[1]
        return new_cnts