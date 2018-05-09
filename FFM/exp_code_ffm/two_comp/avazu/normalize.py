import sys, math

if len(sys.argv) != 3:
    print('usage: cvt.py raw_svm_file normalized_svm_file')
    exit(1)

with open(sys.argv[2], 'w') as f:
    for line in open(sys.argv[1]):
        label, feat_str = line.strip().split(' ', 1)

        feats = []
        two_norm_squaure = 0
        for token in feat_str.split():
            dimension, value = token.split(':')
            dimension = int(dimension)
            value = float(value)
            feats.append((dimension, value))
            two_norm_squaure += value * value

        coef = 1.0 / math.sqrt(two_norm_squaure)

        normalized_feat_tokens = []
        for dimension, value in feats:
            normalized_feat_tokens.append('{0}:{1:.3f}'.format(dimension, coef))

        new_line = label + ' ' + ' '.join(normalized_feat_tokens) + '\n'
        f.write(new_line)
