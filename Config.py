# ==============================================================================
# Copyright (c) 2018, Yamagishi Laboratory, National Institute of Informatics
# Author: Shinji Takaki (takaki@nii.ac.jp)
# All rights reserved.
# ==============================================================================

# directory
prjdir = '.'
datadir = prjdir + '/data'
wavdir = {'trn' : datadir + '/wav_trn',
          'val' : datadir + '/wav_val',
          'test' : datadir + '/wav_test'}
alphadir = {'trn' : None,
            'val' : None,
            'test' : None}
dnames = {'meanstd' : prjdir + '/meanstd',
          'model' : prjdir + '/model',
          'gen' : prjdir + '/gen'}

# speech processing
sf = 16000
fl = 400
fs = 80
fftl = 512
mfbsize = 80

# feedback
fbsize = 400

# training
batch_size = 120
num_frame = 25
lr = 0.001
stop_trigger = (2*10**5, 'iteration')
ext_trigger = (1000, 'iteration')

# gpu
device = 0
