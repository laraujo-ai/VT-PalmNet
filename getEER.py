"""
Compute EER, ROC curve, and DET curve from a matching score file.

Usage:
    python getEER.py  path/to/scores.txt  output_tag

Score file format (one line per pair):
    <distance>  <label>
    where label = 1 (genuine) or -1 (impostor)

Outputs written to path/to/<output_tag>/:
    rst_eer_th_auc.txt   — EER, threshold, AUC
    DET_th_far_frr.txt   — FAR/FRR at each threshold
    ROC.png              — ROC curve (zoomed to FAR < 5%)
    DET.png              — DET curve
    FAR_FRR.png          — FAR and FRR vs threshold
    roc_det.pdf          — combined PDF
"""

import os
import sys
import time

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn import metrics
from sklearn.metrics import auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


if len(sys.argv) < 3:
    print('getEER.py: input args error! using default ...')
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
print('start to load matching scores ...')

pathOut = os.path.join(pathIn, surname)
os.makedirs(pathOut, exist_ok=True)

scores = np.loadtxt(pathScore)

# genuine label == 1, impostor label == -1
inscore  = scores[scores[:, 1] == 1,  0]
outscore = scores[scores[:, 1] == -1, 0]

print('scores loading done!\n')
print('start to calculate EER ...')
start = time.time()
print('numbers of inner & outer matching:')
print(inscore.shape, outscore.shape)

# metrics.roc_curve requires similarity scores (higher = more similar)
# distance scores are lower for matches, so flip if needed
mIn  = inscore.mean()
mOut = outscore.mean()
if mIn < mOut:
    inscore  = -inscore
    outscore = -outscore

y      = np.vstack((np.ones((len(inscore), 1)), np.zeros((len(outscore), 1))))
scores = np.vstack((inscore.reshape(-1, 1), outscore.reshape(-1, 1)))

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
roc_auc = auc(fpr, tpr)

eer    = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

if mIn < mOut:
    thresh     = -thresh
    thresholds = -thresholds

print('eer: %.6f%%  th: %.3f  auc: %.10f' % (eer * 100, thresh, roc_auc))

diffV    = np.abs(fpr - (1 - tpr))
idx      = np.argmin(diffV)
eer_1_2  = (fpr[idx] + (1 - tpr[idx])) / 2.0
th_1_2   = thresholds[idx]
print('eer_1/2: %.6f%%  th_1/2: %.3f  auc: %.10f' % (eer_1_2 * 100, th_1_2, roc_auc))

with open(os.path.join(pathOut, 'rst_eer_th_auc.txt'), 'w') as f:
    f.writelines('%.10f %.4f %.10f\n' % (eer * 100, thresh, roc_auc))
    f.writelines('%.10f %.4f %.10f\n' % (eer_1_2 * 100, th_1_2, -1))

fnr = 1 - tpr
with open(os.path.join(pathOut, 'DET_th_far_frr.txt'), 'w') as f:
    for i in range(len(fpr)):
        f.writelines('%.6f\t%.10f\t%.10f\n' % (thresholds[i], fpr[i], fnr[i]))

pdf = PdfPages(os.path.join(pathOut, 'roc_det.pdf'))

fpr_pct = fpr * 100
tpr_pct = tpr * 100
fnr_pct = fnr * 100

# ROC curve
plt.figure()
plt.plot(fpr_pct, tpr_pct, color='b', linestyle='-', marker='^', label='ROC curve')
plt.plot(np.linspace(0, 100, 101), np.linspace(100, 0, 101), 'k-', label='EER')
plt.xlim([0, 5])
plt.ylim([90, 100])
plt.legend(loc='best')
plt.grid(True)
plt.title('ROC curve')
plt.xlabel('FAR (%)')
plt.ylabel('GAR (%)')
plt.savefig(os.path.join(pathOut, 'ROC.png'))
pdf.savefig()

# DET curve
plt.figure()
plt.plot(fpr_pct, fnr_pct, color='b', linestyle='-', marker='^', label='DET curve')
plt.plot(np.linspace(0, 100, 101), np.linspace(0, 100, 101), 'k-', label='EER')
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.legend(loc='best')
plt.grid(True)
plt.title('DET curve')
plt.xlabel('FAR (%)')
plt.ylabel('FRR (%)')
plt.savefig(os.path.join(pathOut, 'DET.png'))
pdf.savefig()

# FAR / FRR vs threshold
plt.figure()
plt.plot(thresholds, fpr_pct, color='r', linestyle='-', marker='.', label='FAR')
plt.plot(thresholds, fnr_pct, color='b', linestyle='-', marker='^', label='FRR')
plt.legend(loc='best')
plt.grid(True)
plt.title('FAR and FRR Curves')
plt.xlabel('Thresholds')
plt.ylabel('FAR, FRR (%)')
plt.savefig(os.path.join(pathOut, 'FAR_FRR.png'))
pdf.savefig()

pdf.close()
print('done')
