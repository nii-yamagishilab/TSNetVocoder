import cupy, chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, cuda, Variable, reporter
from chainer.functions.array import transpose_sequence
from chainer.links.connection.n_step_rnn import argsort_list_descent, permutate_list

from .spectralloss import spectralloss as lossfn

class Model(Chain):
    def __init__(self, indim, outdim, normfac, fl=400, fs=80, fftl=512, fbsize=400):
        self.indim = indim
        self.outdim = outdim
        self.fl = fl
        self.fs = fs
        self.fftl = fftl
        self.fbsize = fbsize
        self.normfac = {'input'  : {'mean' : cuda.to_gpu(normfac['input']['mean']),
                                    'std' : cupy.fmax(cuda.to_gpu(normfac['input']['std']), 1.0E-6)},
                        'output' : {'mean' : cuda.to_gpu(normfac['output']['mean']),
                                    'std' : cupy.fmax(cuda.to_gpu(normfac['output']['std']), 1.0E-6)}}
        super(Model, self).__init__()
        with self.init_scope():
            self.lx1 = L.NStepBiLSTM(1, self.indim, self.indim//2, 0.0)
            self.lx2 = L.Convolution2D(1, self.indim, (5, self.indim), (1, 1), (2, 0))
            self.ly1 = L.NStepLSTM(3, self.fbsize+self.indim, 256, 0.0)
            self.ly2 = L.Linear(256, self.outdim)

    def _norm(self, X, T):
        if X is not None:
            X = [(x - self.normfac['input']['mean']) / self.normfac['input']['std'] for x in X]
        if T is not None:
            T = [(t - self.normfac['output']['mean']) / self.normfac['output']['std'] for t in T]
        return X, T

    def _unnorm(self, Y):
        Y = [y * self.normfac['output']['std'] + self.normfac['output']['mean'] for y in Y]
        return Y

    def _loss(self, Y, T, A):
        _, T = self._norm(None, T)
        loss = lossfn(F.stack(Y), F.stack(T), F.stack(A), self.fl, self.fftl)
        return loss

    def _forward(self, X, T=None):
        def _InputNet(X):
            H = X
            _, _, H = self.lx1(None, None, H)
            H = [F.reshape(h, (1, 1, -1, self.indim)) for h in H]
            H = [F.tanh(self.lx2(h)) for h in H]
            H = [F.reshape(h.T, (-1, self.indim)) for h in H]
            return H

        def _OutputNet(H, T=None):
            inum = cupy.array(1.0j, cupy.complex64)
            dftnorm = cupy.sqrt(self.fbsize, dtype=cupy.float32)

            if T is not None:
                def _Preprocess(T, H):
                    T = [t.data for t in T]
                    T = [cupy.vstack([cupy.zeros((self.fbsize, t.shape[1]), cupy.float32), t]) for t in T]
                    T = [cupy.hstack([t[i:len(t)-(self.fbsize-i)] for i in range(self.fbsize)]) for t in T]
                    T = [cupy.fft.irfft(cupy.exp(inum*cupy.angle(cupy.fft.rfft(t))))*dftnorm for t in T]
                    T = [Variable(t) for t in T]
                    H = [F.reshape(F.concat(F.broadcast_to(h, (self.fs, h.shape[0], h.shape[1])), axis=1), (h.shape[0]*self.fs, -1)) for h in H]
                    H = [F.concat([t, h]) for t, h in zip(T, H)]
                    return H

                H = _Preprocess(T, H)
                _, _, H = self.ly1(None, None, H)
                Y = [self.ly2(h) for h in H]
            else:
                def _Transpose(X, indices=None, inv=False):
                    if not inv:
                        indices = argsort_list_descent(X)
                        transpose = list(transpose_sequence.transpose_sequence(permutate_list(X, indices, inv=inv)))
                        return indices, transpose
                    else:
                        transpose = permutate_list(transpose_sequence.transpose_sequence(X), indices, inv=inv)
                        return transpose

                def _SampleNet(y, h, hy=None, cy=None):
                    y = Variable(cupy.fft.irfft(cupy.exp(inum*cupy.angle(cupy.fft.rfft(y.data))))*dftnorm)
                    hy = hy[:,:h.shape[0],:] if hy is not None else hy
                    cy = cy[:,:h.shape[0],:] if cy is not None else cy
                    h = F.concat([y, h])
                    h = F.split_axis(h, h.shape[0], axis=0)
                    hy, cy, h = self.ly1(hy, cy, h)
                    h = F.concat(h, axis=0)
                    y = self.ly2(h)
                    return y, hy, cy

                Y = [Variable(cupy.zeros((self.fbsize+len(h)*self.fs, self.outdim), cupy.float32)) for h in H]
                _, H = _Transpose(H)
                indices_y, Y = _Transpose(Y)
                hy, cy = None, None
                y = F.concat([Y[0]] + Y[:self.fbsize-1])
                for i, h in enumerate(H):
                    for j in range(self.fs):
                        y = F.concat([y[:h.shape[0],1:], Y[self.fbsize+i*self.fs+j-1][:h.shape[0]]])
                        Y[self.fbsize+i*self.fs+j], hy, cy = _SampleNet(y, h, hy, cy)
                        Y[self.fbsize+i*self.fs+j].unchain_backward()
                Y = _Transpose(Y, indices_y, inv=True)
                Y = [y[self.fbsize:] for y in Y]
            return Y

        X, T = self._norm(X, T)
        H = _InputNet(X)
        Y = _OutputNet(H, T)
        return Y

    def loss(self, X, T, A):
        def duplication(a):
            return cupy.concatenate(cupy.broadcast_to(a, (self.fs, a.shape[0], a.shape[1])), axis=1).reshape(a.shape[0]*self.fs, -1)

        A = [duplication(a) for a in A]
        X, T = [Variable(x) for x in X], [Variable(t) for t in T]
        Y = self._forward(X, T=T)
        loss, lossA, lossP = self._loss(Y, T, A)
        reporter.report({'loss': loss, 'lossA': lossA, 'lossP': lossP}, self)
        return loss

    def output(self, X):
        with chainer.using_config('train', False):
            X = [Variable(x) for x in X]
            Y = self._forward(X)
            Y = self._unnorm(Y)
            return [cuda.to_cpu(y.data) for y in Y]
