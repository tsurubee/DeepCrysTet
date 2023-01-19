class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._metric = float("inf")
        self.patience = patience
        self.verbose = verbose

    def __call__(self, metric):
        if self._metric < metric:
            self._step += 1
            if self._step >= self.patience:
                if self.verbose:
                    print("early stopping")
                return True
        else:
            self._step = 0
            self._metric = metric
        return False
