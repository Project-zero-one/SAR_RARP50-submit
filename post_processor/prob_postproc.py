import csv
import numpy as np


def read_trans_prob():
    trans_prob = []

    with open('post_processor/trans_prob.csv', mode='r') as f:
        reader = csv.reader(f)
        for i, rows in enumerate(reader):
            if i == 0:
                continue
            trans_prob.append([float(prob) for prob in rows[1:]])
    return trans_prob


class CheckTransProbZero:
    def __init__(self):
        self.trans_prob = read_trans_prob()
        self.prev_class = None

    def __call__(self, prob):
        pred_class = np.argmax(prob)
        result = pred_class

        if self.prev_class is not None:
            trans_prob = self.trans_prob[self.prev_class][pred_class]

            if (pred_class != self.prev_class) & (trans_prob == 0):
                # print(f'prev: {self.prev_class} pred: {pred_class}')
                result = self.prev_class

        self.prev_class = pred_class

        return result
