# TSNetVocoder
- An implementation of TSNetVocoder.
- Paper : Coming soon
- Speech samples : Coming soon

# Requirements
- See Dockerfile.

# Usage
- Wav files need to be put in 'data/wav_trn' (training), 'data/wav_val' (validation) and 'data/wav_test' (analysis-by-synthesis) directories.
  - Following file format is supported.
    - Sampling rate : 16000
    - Quantization bit : 16bit (signed-integer)
    - Number of channels : 1
  - Each utterance should be stored in one wav file.
- By running 00_run.py, you can find a trained model and analysis-by-synthesis wav files in 'model' and 'gen' directories, respectively.
```
python3 00_run.py
```

## Using alpha (Option)
- alphadir written in Config.py need to be modified.
```
alphadir = {'trn' : datadir + '/alpha_trn',
            'val' : datadir + '/alpha_val',
            'test' : None}).
```
- alpha files (format: float, extention: .alpha) need to be put in 'data/alpha_trn' and 'data/alpha_val'.
  - For example, you can use voiced/unvoiced flags as alpha and extract them from speech waveform using SPTK (http://sp-tk.sourceforge.net/) as follows.
```
wav2raw -d ./ hoge.wav
x2x +sf hoge.raw | pitch -p 80 -o 1 | sopr -c 1.0 | interpolate -l 1 -p 257 -d > hoge.alpha
```

#  Who we are
- Shinji Takaki (https://researchmap.jp/takaki/?lang=english)
- Toru Nakashika (http://www.sd.is.uec.ac.jp/nakashika/)
- Xin Wang (https://researchmap.jp/wangxin/?lang=english)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
