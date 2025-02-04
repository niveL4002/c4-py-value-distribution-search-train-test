import math
import random
import json
from typing import Callable

def SE(O: [float], EO: [float]) -> float:
    return sum((o - eo)**2 for o, eo in zip(O, EO))

def d_SE(O: [float], EO: [float]) -> [float]:
    return [2 * (o - eo) for o, eo in zip(O, EO)]

class NN:
    def __init__(self, *, L: [int] = None, filepath: str = None):
        assert (L is None) != (filepath is None)
        if filepath is None:
            self.n = len(L) - 1
            self.L = L[:]
            self.W = [init_weights(L[i], L[i-1]) for i in range(1, len(L))]
            self.B = [[0]*L[i] for i in range(1, len(L))]
        else:
            with open(filepath, "r") as f:
                print("load neural net from", filepath, "...")
                D = json.load(f)
            self.n = D["n"]
            self.L = D["L"]
            self.W = D["W"]
            self.B = D["B"]
        self.Z = [None]*self.n
        self.A = [None]*len(self.L)
        self.dW = [None]*self.n
        self.dB = [None]*self.n
        self.batch_count = 0
        return

    def __str__(self) -> str:
        return str(self.L)

    def compare_info(self, other):
        print("compare net parameters")
        print(self)
        print(other)
        if len(self.L) != len(other.L): return
        if not all(s1 == s2 for s1, s2 in zip(self.L, other.L)): return
        print("biases:")
        for b1, b2 in zip(self.B, other.B):
            assert len(b1) == len(b2)
            db = [v1 - v2 for v1, v2 in zip(b1, b2)]
            mean1 = sum(b1) / len(b1)
            var1 = sum((mean1 - b)**2 for b in b1) / len(b1)
            print("bias len:", len(b1))
            print("self:")
            print("min:", round(min(b1), 4), "max:", round(max(b1), 4),
                  "mean:", round(mean1, 4), "var:", round(var1, 4))
            print("other:")
            mean2 = sum(b2) / len(b2)
            var2 = sum((mean2 - b)**2 for b in b2) / len(b2)
            print("min:", round(min(b2), 4), "max:", round(max(b2), 4),
                  "mean:", round(mean2, 4), "var:", round(var2, 4))
            print("delta (self - other):")
            dmean = sum(db) / len(db)
            dvar = sum((dmean - b)**2 for b in db) / len(db)
            print("min:", round(min(db), 4), "max:", round(max(db), 4),
                  "mean:", round(dmean, 4), "var:", round(dvar, 4))
            print()


    # return reference to internal model output
    def forward(self, I: [float]) -> [float]:
        assert len(I) == self.L[0]
        self.A[0] = I[:]
        for i in range(self.n):
            self.Z[i] = [dot(self.A[i], self.W[i][j]) + self.B[i][j] \
                         for j in range(self.L[i+1])]
            self.A[i+1] = [ReLU(z) for z in self.Z[i]]
        return self.A[-1]

    def zero_gradient(self):
        self.batch_count = 0
        for i in range(self.n):
            self.dW[i] = [[0]*self.L[i] for _ in range(self.L[i+1])]
            self.dB[i] = [0]*self.L[i+1]
        return

    def backward(self, EO: [float], d_loss: Callable = d_SE):
        assert len(EO) == self.L[-1]
        self.batch_count += 1
        G = [d_ReLU(self.Z[-1][i]) * g for i, g in enumerate(d_loss(self.A[-1], EO))]
        i: int = self.n-1
        while True:
            add(self.dB[i], G)
            for j in range(self.L[i+1]):
                add(self.dW[i][j], [a * G[j] for a in self.A[i]])
            if i == 0: break
            i -= 1
            G = [d_ReLU(self.Z[i][j]) * \
                 sum(G[k] * self.W[i+1][k][j] for k in range(self.L[i+2])) \
                 for j in range(self.L[i+1])]
        return

    def update(self, lr: float, mom: float):
        factor: float = lr / self.batch_count
        for i in range(self.n):
            for j in range(self.L[i+1]):
                self.B[i][j] -= factor * self.dB[i][j]
                self.dB[i][j] *= mom
                for k in range(self.L[i]):
                    self.W[i][j][k] -= factor * self.dW[i][j][k]
                    self.dW[i][j][k] *= mom
        self.batch_count = 0
        return

    def train(self, Is: [[float]], EOs: [[float]],
              epochs: int, batch_size, lr: float, mom: float,
              loss: Callable = SE, d_loss: Callable = d_SE) -> None:
        assert len(Is) == len(EOs)
        assert len(Is[0]) == len(self.L[0])
        assert len(EOs[0]) == len(self.L[-1])
        n = len(Is)
        print("n:", n, "batch_size;", batch_size, "lr:", lr, "mom", mom)
        self.zero_gradient()
        indices = list(range(n))
        steps = 0
        for epoch in range(1, epochs+1):
            prin(f"epoch {epoch}/{epochs}:")
            random.shuffle(indices)
            for i in range(batch_size, n+1, batch_size):
                mse = 0
                for j in range(i-batch_size, i):
                    idx = indices[j]
                    out = self.forward(Is[idx])
                    self.backward(EOs[idx])
                    mse += loss(out, EOs[idx])
                mse /= batch_size
                self.update(lr, mom)
                steps += 1
                print(f"{steps}/{epochs*(n // batch_size)}")
        return

    def save(self, filepath: str):
        D = {}
        D["n"] = self.n
        D["L"] = self.L
        D["W"] = self.W
        D["B"] = self.B
        with open(filepath, "w") as f:
            print("save neural net:", self, "to", filepath, "...")
            json.dump(D, f)
        return

def dot(A: [float], B: [float]) -> float:
    assert len(A) == len(B)
    return sum(a*b for a, b in zip(A, B))

def add(A: [float], B: [float]):
    assert len(A) == len(B)
    for i in range(len(A)):
        A[i] += B[i]
    return

def init_weights(n: int, m: int) -> [[float]]:
    return [[random.normalvariate(0, 0.1) for _ in range(m)] for _ in range(n)]

def ReLU(x: float) -> float:
    return x if x > 0 else 0.1 * x

def d_ReLU(x: float) -> float:
    return 1 if x > 0 else 0.1

def test():
    net = NN(L=[2, 2, 1])
    print(net)
    net.save("test_nn")
    nn = NN(filepath="test_nn")
    print(nn)

if __name__ == "__main__":
    test()


