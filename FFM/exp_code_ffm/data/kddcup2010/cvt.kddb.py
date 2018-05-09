#!/usr/bin/env python3

import os, sys
import math
import random
import csv
import collections


test_heading = ['Row', 'Anon Student Id', 'Problem Hierarchy', 'Problem Name', 'Problem View', 'Step Name', 'Step Start Time', 'First Transaction Time', 'Correct Transaction Time', 'Step End Time', 'Step Duration (sec)', 'Correct Step Duration (sec)', 'Error Step Duration (sec)', 'Correct First Attempt', 'Incorrects', 'Hints', 'Corrects', 'KC(SubSkills)', 'Opportunity(SubSkills)', 'KC(KTracedSkills)', 'Opportunity(KTracedSkills)']

test_nonempty_heading = ['Anon Student Id', 'KC(KTracedSkills)', 'KC(SubSkills)', 'Opportunity(KTracedSkills)', 'Opportunity(SubSkills)', 'Problem Hierarchy', 'Problem Name', 'Problem View', 'Step Name']

tr_path = sys.argv[1]
te_path = sys.argv[2]

feat_index_dict = collections.defaultdict(int)

with open(tr_path + '.ffm', 'w') as tr_out, open(tr_path + '.svm', 'w') as tr_svm:
    with open(te_path + '.ffm', 'w') as te_out, open(te_path + '.svm', 'w') as te_svm:
        for index, line in enumerate(csv.DictReader(open(tr_path), delimiter='\t')):
            for key_index, key in enumerate(test_nonempty_heading):
                val = str(key_index) + '---' + line[key]
                if val not in feat_index_dict:
                    feat_index_dict[val] = len(feat_index_dict.keys()) + 1

        for index, line in enumerate(csv.DictReader(open(te_path), delimiter='\t')):
            for key_index, key in enumerate(test_nonempty_heading):
                val = str(key_index) + '---' + line[key]
                if val not in feat_index_dict:
                    feat_index_dict[val] = len(feat_index_dict.keys()) + 1

        val = 1.0/math.sqrt(len(test_nonempty_heading))
        for index, line in enumerate(csv.DictReader(open(tr_path), delimiter='\t')):
            label = line['Correct First Attempt']
            feats = ''
            feats_svm = ''
            for key_index, key in enumerate(test_nonempty_heading):
                feat_val = str(key_index) + '---' + line[key]
                feat_index = feat_index_dict[feat_val]
                feats += ' {0}:{1}:{2:.5f}'.format(key_index + 1, feat_index, val)
                feats_svm += ' {0}:{1:.5f}'.format(feat_index, val)
            tr_out.write('{0} {1}\n'.format(label, feats))
            tr_svm.write('{0} {1}\n'.format(label, feats_svm))

        for index, line in enumerate(csv.DictReader(open(te_path), delimiter='\t')):
            label = line['Correct First Attempt']
            feats = ''
            feats_svm = ''
            for key_index, key in enumerate(test_nonempty_heading):
                feat_val = str(key_index) + '---' + line[key]
                feat_index = feat_index_dict[feat_val]
                feats += ' {0}:{1}:{2:.5f}'.format(key_index + 1, feat_index, val)
                feats_svm += ' {0}:{1:.5f}'.format(feat_index, val)
            te_out.write('{0} {1}\n'.format(label, feats))
            te_svm.write('{0} {1}\n'.format(label, feats_svm))
