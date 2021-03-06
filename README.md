# RIR-Generator-py
为音频添加混响, RIR-Generator 的纯python版本

RIR-Generator: https://github.com/ehabets/RIR-Generator

参考: https://github.com/srikanthrajch/py-RIR-Generator

# 使用

```python
import numpy as np
import rir_generator as RG
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import lfilter

c = 340             # Sound velocity (m/s))
fs = 16000          # Sample frequency (samples/s)
r = [2, 1.5, 2]     # Receiver position [x y z] (m)
s = [2, 3.5, 2]     # Source position [x y z] (m)
L = [5, 4, 6]       # Room dimensions [x y z] (m)
beta = 0.4          # Reverberation time (s)
n = 4096            # Number of samples

h = RG.rir_generator(c, fs, r, s, L, beta=beta, nsample=n)

in_path = 'in.wav'
out_path = 'out.wav'
fs, data = wav.read(in_path)
data = lfilter(h, 1, data)
data = data.astype('int16')
wav.write(out_path, fs, data)
```
