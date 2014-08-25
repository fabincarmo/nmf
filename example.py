#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as pl
import nmf

def main():
    V = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0]])
    r = 2
    W, H = nmf.NMF(V, r, lamb=0, maxit=2000)
    Ve = np.dot(W, H)
    np.set_printoptions(precision=2, suppress=True)

    for i in range(r):
        Hi = np.zeros((r, np.shape(H)[1]))
        Hi[i, :] = H[i, :]
        TMP = np.dot(W, Hi)
        pl.imsave("V"+str(i)+".png", TMP, cmap=pl.cm.gray)

    pl.imsave("V.png", V, cmap=pl.cm.gray)
    pl.imsave("W.png", W, cmap=pl.cm.gray)
    pl.imsave("H.png", H, cmap=pl.cm.gray)
    pl.imsave("Ve.png", Ve, cmap=pl.cm.gray)

    return 0

if __name__ == "__main__":
    main()
