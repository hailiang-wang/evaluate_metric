#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# File: /Users/hain/tmp/evaluate_metric/demo.py
# Author: Hai Liang Wang
# Date: 2017-10-29:15:37:34
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2017-10-29:15:37:34"

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

import matplotlib
matplotlib.use('Agg')
import sklearn.metrics as m
import pylab as pl

class EvaluateCurve(object):
    '''
    draw Evaluate Metrics
    '''
    def __init__(self):
        pass

    def draw_roc_curve(self, lables, scores, name, out = "."):
        #draw roc curve
        fpr, tpr, th = m.roc_curve(lables, scores)
        roc_auc = round(m.auc(fpr, tpr), 5)
        name = name + '  auc=' + str(roc_auc)
        pl.plot(fpr, tpr, label=name)
        pl.xlabel('FPR')
        pl.ylabel('TPR')
        pl.legend(loc=4)
        pl.title("Roc Curve")
        pl.savefig(os.path.join(out,'roc_curve.png'), format='png')
        pl.gcf().clear()

    def draw_pr_with_hresholds(self, lables, scores, name="", out = "."):
        '''
        draw PR with hresholds
        '''
        precision, recall, threshold = m.precision_recall_curve(lables, scores)
        pl.plot(threshold, precision[:-1], "b--", label="Precision")
        pl.plot(threshold, recall[:-1], "g-", label="Recall")
        pl.xlabel("Threshold")
        pl.legend(loc=4)
        pl.savefig(os.path.join(out,'pr_threshold.png'), format='png')
        pl.gcf().clear()

def main():
    scores = []
    lables = []

    with open(os.path.join(curdir, "sim.txt"), "r") as _fin:
        for x in _fin.readlines():
            lables.append(float(x.strip()))

    with open(os.path.join(curdir, "output_euclidean_stopwords_nearby_sum.txt"), 'r') as _fin:
        for x in _fin.readlines():
            items = x.split(":")
            if len(items) != 3: continue
            score = float(items[1].strip())
            scores.append(score)

    assert len(scores) == len(lables), "wrong labeling"
    eval_output = EvaluateCurve()
    eval_output.draw_pr_with_hresholds(lables, scores, "pr_evaluation")

if __name__ == '__main__':
    main()
