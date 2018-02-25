import math
import numpy as np
import matplotlib.pyplot as plt

class DemoGenerator:
    def __init__(self):
        self.MIN_LENGTH = 500
    
    def mysin(self, length, step=0.05):
        stt = np.random.rand() * math.pi
        return np.sin(stt + np.arange(length) * step)
    
    def myrnd(self, length, scale=0.3):
        return np.random.randn(length) * scale
    
    def _generate(self, length, category):
        assert category == 0 or category == 1
        if category == 0:
            return self.mysin(length)
        elif category == 1:
            return self.myrnd(length)
    
    def generate(self, size):
        """
        :param size:how many test waveforms to generate
        :ret a pair of test waveforms and label arraies
        """
        
        x_lst = []
        y_lst = []
        
        for _ in range(size):
            x, y = [], []
            ctg = np.random.randint(2) % 2 # 0 or 1
            while len(x) < self.MIN_LENGTH:
                length = np.random.randint(50, 200)
                x += list(self._generate(length, ctg))
                y += list(np.ones(length) * ctg)
                ctg = (ctg + 1) % 2
            assert len(x) == len(y)
            x_lst.append(np.array(x))
            y_lst.append(np.array(y).astype(int))
        
        return x_lst, y_lst
    
def myplot(x, y):
    assert len(x) == len(y)
    length = len(x)
    colors = ("tomato", "royalblue")
    
    prv = 0
    ctg = y[0]
    for idx in range(length):
        if y[idx] != ctg:
            plt.plot(np.arange(prv, idx), x[prv:idx], colors[ctg])
            prv = idx
            ctg = y[idx]
    plt.plot(np.arange(prv, length), x[prv:length], colors[ctg])
    plt.show()