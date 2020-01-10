import time

import numpy as np


class Logger:

    def __init__(self):
        self.train_loss = []
        self.train_conf = []

        self.valid_loss = []
        self.valid_conf = []

        self.test_loss = []
        self.test_conf = []

        self.elapsed_time = []

        self.training_time = []

        self.init_time = time.time()
        self.eval_start_time = time.time()
        self.eval_time = 0

        self.best_valid_model = None

    def print_info(self, iter_idx, start_time):
        print(
            'Iter {:<4} - time: {:<5} - [train] loss: {:<6} (+/-{:<6}) - [valid] loss: {:<6} (+/-{:<6}) - [test] loss: {:<6} (+/-{:<6})'.format(
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.train_conf[-1], 4),
                np.round(self.valid_loss[-1], 4),
                np.round(self.valid_conf[-1], 4),
                np.round(self.test_loss[-1], 4),
                np.round(self.test_conf[-1], 4),
            )
        )

    def track_time(self):
        self.elapsed_time.append(time.time() - self.init_time)
        self.eval_time += time.time() - self.eval_start_time
        self.training_time.append((time.time() - self.init_time) - self.eval_time)

    def time_eval(self):
        self.eval_start_time = time.time()
