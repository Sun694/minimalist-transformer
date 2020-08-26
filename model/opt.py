# class inspired by the annotated transformer
# http://nlp.seas.harvard.edu/2018/04/03


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        Optimizer wrapper.
        Args:
            model_size: d_model of transformer
            factor: coefficient of rate
            warmup: number of warmup steps
            optimizer: pytorch optimizer

        default setup:
        NoamOpt(embedding_dim, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-8)
        """
        self.d_model = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self.num_steps = 0
        self.rate = 0

    def step(self):
        """ single optimization step """
        self.num_steps += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.rate = rate
        self.optimizer.step()

    def get_rate(self, step=None):
        """ gets the rate, part of warmup """
        if step is None:
            step = self.num_steps
        return self.factor * (
            self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()
