# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Shinji Takaki (takaki@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import cupy
from chainer import function_node, functions
from chainer.utils import type_check

class SpectralLoss(function_node.FunctionNode):
    def __init__(self, alpha, fl=400, fftl=512):
        super(SpectralLoss, self).__init__()
        self.alpha = alpha[:,fl//2-1:-fl//2]
        self.fl = fl
        self.fftl = fftl
        self.norm = cupy.sqrt(self.fftl, dtype=cupy.float32)

    def _frame(self, X):
        F = cupy.stack([X[:,i:len(X[0])-(self.fl-1-i)] for i in range(self.fl)], axis=2).reshape(len(X), -1, self.fl)
        return F

    def _overlapadd(self, F):
        N = len(F)
        F = F[:,:,:self.fl]
        X = cupy.sum(cupy.stack([cupy.hstack([cupy.zeros((N, i, 1), cupy.float32), F[:,:,i:i+1], cupy.zeros((N, self.fl-1-i, 1), cupy.float32)]) for i in range(self.fl)]), axis=0)
        return X

    def _spectrum(self, x, y):
        S = {'x' : cupy.fft.rfft(self._frame(x), self.fftl) / self.norm,
             'y' : cupy.fft.rfft(self._frame(y), self.fftl) / self.norm}
        A = {'x' : cupy.abs(S['x']),
             'y' : cupy.abs(S['y'])}
        P = {'x' : cupy.angle(S['x']),
             'y' : cupy.angle(S['y'])}
        return A, P

    def _lossA(self, A, P):
        inum = cupy.array(1.0j, cupy.complex64)
        loss = cupy.square(A['y']-A['x']).mean()
        grad = self._overlapadd(cupy.fft.irfft((A['x']-A['y']) * cupy.exp(inum*P['x'])) * self.norm)
        return loss, grad

    def _lossP(self, A, P):
        floor = 1.0E-6
        inum = cupy.array(1.0j, cupy.complex64)
        loss = (self.alpha * (1 - cupy.cos(P['y']-P['x']))).mean()
        grad = self._overlapadd(cupy.fft.irfft(self.alpha * cupy.sin(P['y']-P['x']) / cupy.fmax(A['x'], floor) * cupy.exp(inum*(P['x']-0.5*cupy.pi))) * self.norm)
        return loss, grad

    def _loss(self, x, y):
        A, P = self._spectrum(x, y)
        lossA, gradA = self._lossA(A, P)
        lossP, gradP = self._lossP(A, P)
        return {'A': lossA, 'P': lossP}, {'A': gradA, 'P': gradP}

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(in_types[0].dtype == cupy.float32, in_types[1].dtype == cupy.float32, in_types[0].shape == in_types[1].shape,)

    def forward(self, inputs):
        self.retain_outputs((3,))
        loss, grad = self._loss(inputs[0], inputs[1])
        return loss['A']+loss['P'], loss['A'], loss['P'], grad['A']+grad['P']

    def backward(self, indexes, gy):
        grad = self.get_retained_outputs()[0]
        gy0 = functions.broadcast_to(gy[0], grad.shape)
        gx0 = gy0 * grad / grad.size
        return gx0, -gx0

def spectralloss(x0, x1, alpha, fl, fftl):
    return SpectralLoss(alpha.data, fl, fftl).apply((x0, x1))[0:3]
