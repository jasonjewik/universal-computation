import torch
from tqdm import tqdm

import time
from collections import defaultdict


class Trainer:

    def __init__(
            self,
            model,
            dataset,
            loss_fn,
            perf_metrics=None,
            steps_per_epoch=100,
            test_steps_per_epoch=20,
            learning_rate=1e-3,
            batch_size=2,
            eval_batch_size=8,
            grad_accumulate=1,
    ):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.perf_metrics = perf_metrics
        self.steps_per_epoch = steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.grad_accumulate = grad_accumulate

        self.optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.diagnostics = {'Gradient Steps': 0}

    def get_loss(self, x, y, return_perf=False):
        out = self.model(x)
        loss = self.loss_fn(out, y, x=x)
        perfs = {}
        if return_perf:
            if self.perf_metrics is None:
                raise NotImplementedError('performance metrics not specified')
            for pm in self.perf_metrics:
                perfs[pm.name] = pm(
                    out.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    x=x.detach().cpu().numpy(),
                )
            return loss, perfs
        return loss

    def train_epoch(self, test_steps=None):
        self.dataset.start_epoch()

        train_losses, tr_perfs = [], defaultdict(float)
        self.model.train()
        start_train_time = time.time()
        for _ in tqdm(range(self.steps_per_epoch)):
            step_loss = 0
            for _ in range(self.grad_accumulate):
                x, y = self.dataset.get_batch(self.batch_size, train=True)
                loss, perfs = self.get_loss(x, y, return_perf=True)
                loss = loss / self.grad_accumulate
                loss.backward()
                step_loss += loss.detach().cpu().item()
                for key, val in perfs.items():
                    tr_perfs[key] += val

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optim.step()
            self.optim.zero_grad()

            self.diagnostics['Gradient Steps'] += 1

            train_losses.append(step_loss)
        end_train_time = time.time()

        test_steps = self.test_steps_per_epoch if test_steps is None else test_steps

        test_loss, performance = 0., defaultdict(int)
        self.model.eval()
        start_test_time = time.time()
        with torch.no_grad():
            for _ in range(test_steps):
                x, y = self.dataset.get_batch(self.eval_batch_size, train=False)
                loss, perf = self.get_loss(x, y, return_perf=True)
                test_loss += loss.detach().cpu().item() / test_steps
                for key in performance:
                    performance[key] += perf[key] / test_steps
        end_test_time = time.time()

        self.diagnostics['Average Train Loss'] = sum(train_losses) / self.steps_per_epoch
        self.diagnostics['Start Train Loss'] = train_losses[0]
        self.diagnostics['Final Train Loss'] = train_losses[-1]
        self.diagnostics['Test Loss'] = test_loss
        for key, val in performance.items():
            self.diagnostics[f'Test {key}'] = val
        for key, val in tr_perfs.items():
            self.diagnostics[f'Train {key}'] = val / (self.steps_per_epoch * self.grad_accumulate)
        self.diagnostics['Time Training'] = end_train_time - start_train_time
        self.diagnostics['Time Testing'] = end_test_time - start_test_time
