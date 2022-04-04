class PerfMetric():

    def __init__(
            self,
            name,
            func
    ):
        self.name = name
        self.func = func

    def __call__(self, preds, true, x=None):
        return self.func(preds, true, x)
