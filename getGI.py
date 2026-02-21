"""
Plot the Genuine / Impostor score distribution histogram.

Usage:
    python getGI.py  path/to/scores.txt  output_tag

Score file format (one line per pair):
    <distance>  <label>
    where label = 1 (genuine) or -1 (impostor)

Outputs written to path/to/<output_tag>/:
    GI_curve.png             — genuine vs impostor histogram
    matching_score_distr.txt — min/max/mean/std for each class
    matching_hist.txt        — raw histogram values
"""

import os
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.switch_backend('agg')


if len(sys.argv) < 3:
    print('getGI.py: input args error! using default ...')
    pathScore = './scores.txt'
    surname   = 'scores'
else:
    pathScore = sys.argv[1]
    surname   = sys.argv[2]

pathIn    = os.path.dirname(pathScore)
scorename = os.path.basename(pathScore)

print('pathIn: ',   pathIn)
print('scorename:', scorename)
print('surname:',   surname)
print('start to load matching scores ...\n')

pathOut = os.path.join(pathIn, surname)
os.makedirs(pathOut, exist_ok=True)

scores   = np.loadtxt(pathScore)
inscore  = scores[scores[:, 1] == 1,  0]
outscore = scores[scores[:, 1] == -1, 0]

print('inner  (min, max, mean, std): [%f, %f] [%f +- %f]'
      % (inscore.min(), inscore.max(), inscore.mean(), inscore.std()))
print('outer  (min, max, mean, std): [%f, %f] [%f +- %f]'
      % (outscore.min(), outscore.max(), outscore.mean(), outscore.std()))
print('scores loading done! start to plot histograms ...')

maxvin  = inscore.max()
minvin  = inscore.min()
maxvo   = outscore.max()
minvo   = outscore.min()

meanvin = inscore.mean()
stdvin  = inscore.std()
meanvo  = outscore.mean()
stdvo   = outscore.std()

samples = 100

inscore_n  = (inscore  - minvin) / (maxvin - minvin) * samples
outscore_n = (outscore - minvo)  / (maxvo  - minvo)  * samples

histin = np.zeros(samples + 1, dtype='int32')
histo  = np.zeros(samples + 1, dtype='int32')

for i in inscore_n:
    histin[int(round(i))] += 1
for i in outscore_n:
    histo[int(round(i))] += 1

histin = histin / histin.sum() * 100
histo  = histo  / histo.sum()  * 100

plt.figure(1)
plt.plot(np.linspace(0, 1, samples + 1) * (maxvo - minvo) + minvo, histo,  'r', label='Impostor')
plt.plot(np.linspace(0, 1, samples + 1) * (maxvin - minvin) + minvin, histin, 'b', label='Genuine')
plt.legend(loc='upper right', fontsize=13)
plt.xlabel('Matching Score', fontsize=13)
plt.ylabel('Percentage (%)', fontsize=13)
plt.ylim([0, 1.2 * max(histin.max(), histo.max())])
plt.grid(True)
plt.savefig(os.path.join(pathOut, 'GI_curve.png'))

with open(os.path.join(pathOut, 'matching_score_distr.txt'), 'w') as f:
    f.writelines('[min, max] [mean +- std]\n')
    f.writelines('inner: [%.10f, %.10f] [%.10f +- %.10f]\n' % (minvin, maxvin, meanvin, stdvin))
    f.writelines('outer: [%.10f, %.10f] [%.10f +- %.10f]\n' % (minvo,  maxvo,  meanvo,  stdvo))
    f.writelines('number of genuine matching:  %d\n' % len(inscore))
    f.writelines('number of impostor matching: %d\n' % len(outscore))

xin = np.linspace(0, 1, samples + 1) * (maxvin - minvin) + minvin
xo  = np.linspace(0, 1, samples + 1) * (maxvo  - minvo)  + minvo

with open(os.path.join(pathOut, 'matching_hist.txt'), 'w') as f:
    for i in range(samples + 1):
        f.writelines('%.4f %.4f %.4f %.4f\n' % (xin[i], histin[i], xo[i], histo[i]))

print('done!\n')
