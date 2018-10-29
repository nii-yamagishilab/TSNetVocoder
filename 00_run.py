#!/usr/bin/python3
# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Shinji Takaki (takaki@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import pathlib, wave, numpy

import matplotlib, collections
from chainer import cuda, optimizers, serializers
from chainer.iterators import MultithreadIterator as dataiter
from chainer.datasets import TransformDataset as datatf
from chainer.training import updaters, Trainer, extensions
from chainer.dataset.convert import Conveyor

from module import Model, SpeechProcessing
import Config as cfg

matplotlib.use('Agg')

def main():
    class dataprocess(object):
        def __init__(self, sp, num_frame=None):
            self.sp = sp
            self.num_frame = num_frame
            self.margin_frame = (sp.fl // 2) // sp.fs + 1

        def __call__(self, args):
            def loadwav(fname):
                with wave.Wave_read(fname) as f:
                    assert f.getframerate() == sp.sf, 'sampling rate is different ('+fname+')'
                    assert f.getnchannels() == 1, 'channel is not 1 ('+fname+')'
                    T = numpy.frombuffer(f.readframes(f.getnframes()), numpy.int16).astype(numpy.float32)
                    return T

            def loadalpha(fname, size):
                if fname is not None:
                    A = numpy.fromfile(open(fname, 'rb'), numpy.float32).reshape(-1, self.sp.fftl//2+1)
                else:
                    A = numpy.ones((size, self.sp.fftl//2+1), numpy.float32)
                return A

            def segment(T, A):
                if self.num_frame is not None:
                    l = (self.num_frame + 2 * self.margin_frame) * self.sp.fs
                    s = numpy.random.randint(len(T) - l)
                    T = T[s:s+l]
                    s = numpy.minimum(s//self.sp.fs, len(A) - self.num_frame - 2 * self.margin_frame)
                    A = A[s:s+self.num_frame+2*self.margin_frame]
                return T, A

            def trim_margin(X, T, A):
                if self.num_frame is not None:
                    X = X[self.margin_frame:self.margin_frame+self.num_frame]
                    T = T[self.margin_frame*self.sp.fs-self.sp.fs//2:(self.margin_frame+self.num_frame)*self.sp.fs-self.sp.fs//2]
                    A = A[self.margin_frame:self.margin_frame+self.num_frame]
                return X, T, A

            wavfname, alphafname = args[0], args[1]
            T = loadwav(wavfname)
            A = loadalpha(alphafname, len(T)//self.sp.fs)
            T, A = segment(T, A)
            X = self.sp.analyze(T)
            X, T, A = trim_margin(X, T, A)
            return X, T.reshape(-1, 1), A

    class dataconverter(object):
        def __init__(self, stream=None):
            self._stream = stream
            self._device = None
            self._conveyor = collections.defaultdict(lambda: Conveyor(self._device, self._stream))

        def __call__(self, batch, device=None):
            assert len(batch) != 0, 'batch is empty'
            first_elem = batch[0]
            if len(self._conveyor) == 0:
                self._device = device
                if device is not None and device >= 0 and self._stream is None:
                    with cuda.get_device_from_id(device):
                        self._stream = cuda.Stream(non_blocking=True)
            assert device is self._device, 'device is different'
            with cuda.get_device_from_id(device):
                if isinstance(first_elem, tuple):
                    I, J = len(first_elem), len(batch)
                    result = [[] for i in range(I)]
                    for i in range(I):
                        for j in range(J):
                            self._conveyor[j*J+i].put(batch[j][i])
                    for i in range(I):
                        for j in range(J):
                            result[i].append(self._conveyor[j*J+i].get())
                    return tuple(result)
            assert False, 'Not supported'

    def randomseed():
        numpy.random.seed(123)
        cuda.cupy.random.seed(321)

    def makedirs(dirs):
        for d in dirs.values():
            pathlib.Path(d).mkdir(exist_ok=True)

    def getmeanstd(msdir, F, sp):
        fnames = {'input'  : {'mean' : msdir + '/input.mean',
                              'std'  : msdir + '/input.std'},
                  'output' : {'mean' : msdir + '/output.mean',
                              'std'  : msdir + '/output.std'}}
        K = (('input', 'mean'), ('input', 'std'), ('output', 'mean'), ('output', 'std'))

        if all([pathlib.Path(fnames[k0][k1]).exists() for k0, k1 in K]):
            print('Mean and standard deviation load')
            ms = {'input'  : {'mean' : numpy.frombuffer(open(fnames['input']['mean'], 'rb').read(), numpy.float32),
                              'std'  : numpy.frombuffer(open(fnames['input']['std'], 'rb').read(), numpy.float32)},
                  'output' : {'mean' : numpy.frombuffer(open(fnames['output']['mean'], 'rb').read(), numpy.float32),
                              'std'  : numpy.frombuffer(open(fnames['output']['std'], 'rb').read(), numpy.float32)}}
        else:
            b = dataiter(datatf(F, dataprocess(sp)), len(F), repeat=False).next()
            X, T = numpy.vstack([d[0] for d in b]), numpy.vstack([d[1] for d in b])
            ms = {'input'  : {'mean' : numpy.mean(X, axis=0),
                              'std'  : numpy.std(X, axis=0)},
                  'output' : {'mean' : numpy.mean(T, axis=0),
                              'std'  : numpy.std(T, axis=0)}}
            for k0, k1 in K:
                ms[k0][k1].tofile(fnames[k0][k1])
        return ms

    def getfnames(wavdir, alphadir):
        F = {}
        for i in ('trn', 'val', 'test'):
            F[i] = []
            for stem in [p.stem for p in pathlib.Path(wavdir[i]).glob('*.wav')]:
                wavfname = wavdir[i] + '/' + stem + '.wav'
                alphafname = alphadir[i] + '/' + stem + '.alpha' if alphadir[i] is not None else None
                F[i].append((wavfname, alphafname))
        return F

    def initmodel(indim, outdim, normfac, fl, fs, fftl, fbsize, device):
        model = Model(indim, outdim, normfac, fl, fs, fftl, fbsize)
        model.to_gpu(device)
        return model

    def training(modeldir, F, sp, model, batch_size, num_frame, lr, stop_trigger, ext_trigger, device):
        B = {'trn' : dataiter(datatf(F['trn'], dataprocess(sp, num_frame)), batch_size),
             'val' : dataiter(datatf(F['val'], dataprocess(sp, num_frame)), batch_size, False, False)}

        optimizer = optimizers.Adam(lr)
        optimizer.setup(model)
        updater = updaters.StandardUpdater(B['trn'], optimizer, converter=dataconverter(), loss_func=model.loss, device=device)
        trainer = Trainer(updater, stop_trigger, out=modeldir)
        trainer.extend(extensions.Evaluator(B['val'], model, eval_func=model.loss, device=device), trigger=ext_trigger)
        trainer.extend(extensions.snapshot(), trigger=ext_trigger)
        trainer.extend(extensions.LogReport(trigger=ext_trigger))
        trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/lossA', 'validation/main/lossA', 'main/lossP', 'validation/main/lossP']))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'iteration', file_name='loss.png', trigger=ext_trigger))
        trainer.extend(extensions.PlotReport(['main/lossA', 'validation/main/lossA'], 'iteration', file_name='lossA.png', trigger=ext_trigger))
        trainer.extend(extensions.PlotReport(['main/lossP', 'validation/main/lossP'], 'iteration', file_name='lossP.png', trigger=ext_trigger))

        ss = sorted([int(str(p).split('_')[-1]) for p in pathlib.Path(modeldir).glob('snapshot_iter_*')])
        if ss:
            print('Trainer load : snapshot_iter_'+str(ss[-1]))
            serializers.load_npz(modeldir+'/snapshot_iter_'+str(ss[-1]), trainer)

        trainer.run()
        return model

    def synthesis(gendir, F, sp, model, device):
        def write_wav(fname, y):
            with wave.Wave_write(fname) as f:
                f.setparams((1, 2, cfg.sf, len(y), "NONE", "not compressed"))
                f.writeframes(y.astype(numpy.int16))

        batch_size = len(F)
        B = dataiter(datatf(F, dataprocess(sp)), batch_size, False, False)
        converter = dataconverter()

        for i, b in enumerate(B):
            S = [pathlib.PurePath(f[0]).stem for f in F[i*batch_size:(i+1)*batch_size]]
            X, _, _ = converter(b, device)
            for y, s in zip(model.output(X), S):
                write_wav(gendir+'/'+s+'.wav', y)

    cuda.get_device(cfg.device).use()
    sp = SpeechProcessing(cfg.sf, cfg.fl, cfg.fs, cfg.fftl, cfg.mfbsize)
    F = getfnames(cfg.wavdir, cfg.alphadir)
    randomseed()

    print('--- Directory making ---')
    makedirs(cfg.dnames)

    print('--- Mean and standard deviation calculation ---')
    ms = getmeanstd(cfg.dnames['meanstd'], F['trn'], sp)

    print('--- Model initialization  ---')
    model = initmodel(sp.mfbsize, 1, ms, sp.fl, sp.fs, sp.fftl, cfg.fbsize, cfg.device)

    print('--- Model training ---')
    model = training(cfg.dnames['model'], F, sp, model, cfg.batch_size, cfg.num_frame, cfg.lr, cfg.stop_trigger, cfg.ext_trigger, cfg.device)

    print('--- Synthesis ---')
    synthesis(cfg.dnames['gen'], F['test'], sp, model, cfg.device)

    print('Done')

if __name__ == '__main__':
    main()
