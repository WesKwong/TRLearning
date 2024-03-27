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

class no_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts):
        super().__init__(ngram_cnts, context_cnts)

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram]
        divisor = self.context_cnts[context]
        if divisor == 0:
            divisor = 1
        prob = dividend / divisor
        return prob


class add1_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts, vocab_size):
        super().__init__(ngram_cnts, context_cnts)
        self.vocab_size = vocab_size

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram] + 1
        divisor = self.context_cnts[context] + self.vocab_size
        if divisor == 0:
            divisor = 1
        prob = dividend / divisor
        return prob

class gt_smooth(ProbEstimator):
    def __init__(self, ngram_cnts, context_cnts, threshold=9):
        super().__init__(ngram_cnts, context_cnts)
        self.ngram_cnts = self.get_new_cnts(ngram_cnts, threshold)
        self.context_cnts = self.get_new_cnts(context_cnts, threshold)

    def __call__(self, ngram):
        ngram = tuple(ngram)
        context = ngram[:-1]
        dividend = self.ngram_cnts[ngram] if self.ngram_cnts[ngram] else self.ngram_cnts['<ZERO>']
        divisor = self.context_cnts[context] if self.context_cnts[context] else self.context_cnts['<ZERO>']
        prob = dividend / divisor
        return prob

    def get_new_cnts(self, cnts, threshold):
        from collections import Counter
        import copy
        # get dr
        N = sum(cnts.values())
        Nr = Counter(cnts.values())
        dr = Counter()
        for r in range(1, threshold+1):
            dr[r] = (r+1) * Nr[r+1] / Nr[r]
            dr[r] -= r * (threshold+1) * Nr[threshold+1] / Nr[1]
            dr[r] /= 1 - (threshold+1) * Nr[threshold+1] / Nr[1]
        # get new cnts
        new_cnts = copy.deepcopy(cnts)
        for key, value in new_cnts.items():
            if value <= threshold:
                new_cnts[key] = dr[value]
        new_cnts['<ZERO>'] = Nr[1]
        return new_cnts