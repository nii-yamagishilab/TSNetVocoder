# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Shinji Takaki (takaki@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import numpy

class SpeechProcessing(object):
    def __init__(self, sf=16000, fl=400, fs=80, fftl=512, mfbsize=80):
        self.sf = sf
        self.fl = fl
        self.fs = fs
        self.fftl = fftl
        self.mfbsize = mfbsize
        winpower = numpy.sqrt(numpy.sum(numpy.square(numpy.blackman(self.fl).astype(numpy.float32))))
        self.window = numpy.blackman(self.fl).astype(numpy.float32) / winpower
        self.melfb = self._melfbank()

    def _freq2mel(self, freq):
        return 1127.01048 * numpy.log(freq / 700.0 + 1.0)

    def _mel2freq(self, mel):
        return (numpy.exp(mel / 1127.01048) - 1.0) * 700.0
        
    def _melfbank(self):
        linear_freq = 1000.0
        mfbsize = self.mfbsize - 1

        bFreq = numpy.linspace(0, self.sf / 2.0, self.fftl//2 + 1, dtype=numpy.float32)
        minMel = self._freq2mel(0.0)
        maxMel = self._freq2mel(self.sf / 2.0)
        iFreq = self._mel2freq(numpy.linspace(minMel, maxMel, mfbsize + 2, dtype=numpy.float32))
        linear_dim = numpy.where(iFreq<linear_freq)[0].size
        iFreq[:linear_dim+1] = numpy.linspace(iFreq[0], iFreq[linear_dim], linear_dim+1)

        diff = numpy.diff(iFreq)
        so = numpy.subtract.outer(iFreq, bFreq)
        lower = -so[:mfbsize] / numpy.expand_dims(diff[:mfbsize], 1)
        upper = so[2:] / numpy.expand_dims(diff[1:], 1)
        fb = numpy.maximum(0, numpy.minimum(lower, upper))

        enorm = 2.0 / (iFreq[2:mfbsize+2] - iFreq[:mfbsize])
        fb *= enorm[:, numpy.newaxis]

        fb0 = numpy.hstack([numpy.array(2.0*(self.fftl//2)/self.sf, numpy.float32), numpy.zeros(self.fftl//2, numpy.float32)])
        fb = numpy.vstack([fb0, fb])

        return fb

    def _frame(self, X):
        X = numpy.concatenate([numpy.zeros(self.fl//2, numpy.float32), X, numpy.zeros(self.fl//2, numpy.float32)])
        X = X[:(len(X)-self.fl-1)//self.fs*self.fs+self.fl].reshape(-1, self.fs)
        F = numpy.hstack([X[i:len(X)-(self.fl//self.fs-1-i)] for i in range(self.fl//self.fs)])
        return F

    def _anawindow(self, F):
        W = F * self.window
        return W

    def _rfft(self, W):
        Y = numpy.fft.rfft(W, n=self.fftl).astype(numpy.complex64)
        return Y

    def _amplitude(self, Y):
        eps = 1.0E-12
        A = numpy.fmax(numpy.absolute(Y), eps)
        return A

    def _logmelfbspec(self, A):
        M = numpy.log(numpy.dot(A, self.melfb.T))
        return M

    def analyze(self, X):
        M = self._logmelfbspec(self._amplitude(self._rfft(self._anawindow(self._frame(X)))))
        return M
