from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
np.random.seed(1)

word_to_ix = {"hello": 0, "world":1}
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

"""
1. N-Gram Language Model
"""
print("N-Gram Language Model")

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# Build tuples. Each tuple is ([word_i - 2, word_i-1], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])  for i in range(len(test_sentence)-2)]
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = { word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, *input):
        embeds = self.embedding(input[0]).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:

        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        model.zero_grad()

        log_probs = model(context_idxs)

        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        loss.backward()
        optimizer.step()

        total_loss += loss
    losses.append(total_loss)
print(losses)


"""
2. Continuous Bag Of Words
"""
print("CBOW")

CONTEXT_SIZE = 2 # 2 words left, 2 words right

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split(' ')

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1],
               raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOWNaive(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOWNaive, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, *input):
        # 输入已经是context的id列表了
        embeds = self.embedding(input[0]).mean(dim=0).view(1, -1)

        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)
# make_context_vector(data[0][0], word_to_ix)

# Negative Log Likelihood Loss
loss_function = nn.NLLLoss()
model = CBOWNaive(vocab_size, EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []

for epoch in range(10):
    total_loss = 0.
    for context, target in data:
        context_ixs = make_context_vector(context, word_to_ix)
        target_ix = torch.tensor([word_to_ix[target]], dtype=torch.long)

        model.zero_grad()

        log_probs = model(context_ixs)
        loss = loss_function(log_probs, target_ix)

        loss.backward()
        optimizer.step()

        total_loss += loss
    losses.append(total_loss)
print(losses)


"""
3. SkipGram
"""
print("SkipGram Naive sample")
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_data = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_data)
vocab_size = len(vocab)

word_to_ix = {word:i for i, word in enumerate(vocab)}


data = []
for ix in range(CONTEXT_SIZE, len(raw_data) - CONTEXT_SIZE):
    center = raw_data[ix]
    context = []
    for w in range(ix-CONTEXT_SIZE, ix + CONTEXT_SIZE + 1):
        if w == ix:
            continue
        context.append(raw_data[w])
    data.append((context, center))

print(data[:5])

class SkipGramNaiveModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):
        super(SkipGramNaiveModel, self).__init__()
        self.embedding_in = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.context_size = context_size

    def forward(self, *input):
        embeds = self.embedding_in(input[0]).view(1, -1)

        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


loss_function = nn.NLLLoss()
model = SkipGramNaiveModel(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def make_context_ixs(context, word_to_ix):
    context = [word_to_ix[word] for word in context]
    return torch.tensor(context, dtype=torch.long)

losses = []
for epoch in range(10):
    total_loss = 0
    for context, center in data:
        context_ixs = make_context_ixs(context, word_to_ix)
        center_ix = torch.tensor([word_to_ix[center]], dtype=torch.long)

        center_ixs = center_ix.repeat(CONTEXT_SIZE)

        for center, context in zip(center_ixs, context_ixs):
            model.zero_grad()

            log_probs = model(center.view(1))
            loss = loss_function(log_probs, context.view(1))

            loss.backward()
            optimizer.step()
            total_loss += loss
    losses.append(total_loss)
print(losses)

"""
Skip-Gram Negative Sample
"""
print("Skip-gram negative sample")

# Subsample threshold
T = 1e-5
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

class PermutedSubsampleedCorpus(Dataset):

    def __init__(self, data, word_sample):
        self.data = []
        for iword, owords in data:
            if np.random.rand() > word_sample[iword]:
                self.data.append((iword, owords))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        iword, owords = self.data[item]
        # 按列拼接形成batch的
        return (iword, owords)


class SkipGramNegativeSample(nn.Module):

    def __init__(self, vocab_size, embedding_size, n_negs):
        super(SkipGramNegativeSample, self).__init__()
        self.ivectors = nn.Embedding(vocab_size, embedding_size)
        self.ovectors = nn.Embedding(vocab_size, embedding_size)

        self.ivectors.weight.data.uniform_(- 0.5/embedding_size, 0.5/embedding_size)
        self.ovectors.weight.data.zero_()

        self.n_negs = n_negs
        self.vocab_size = vocab_size

    def forward(self, iwords, owords):
        # iwords: (batch_size)
        # owords: (batch_size, context_size * 2)
        batch_size = iwords.size()[0]
        context_size = owords.size()[-1] # 两边的context之和

        nwords = torch.FloatTensor(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()

        ivectors = self.ivectors(iwords).unsqueeze(2) # (batch_size, embeding_dim, 1)
        ovectors = self.ovectors(owords) # (batch_size, context_size, embedding_dim)
        nvectors = self.ovectors(nwords).neg() #(batch_size, context_size * n_negs, embedding_dim)

        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean() #(batch_size)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1) #(batch_size)
        return -(oloss + nloss).mean()

f = open("./text9", "rb+")
raw_data = f.readlines()
f.close()

raw_data = raw_data[0].split()

vocab = set(raw_data)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

word_count = dict()
for word in raw_data:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 0
# print(word_count)


word_frequency = np.array(list(word_count.values()))
word_frequency = word_frequency / word_frequency.sum()
word_sample = 1 - np.sqrt(T / word_frequency)
word_sample = np.clip(word_sample, 0, 1)
word_sample = {wc[0]: s for wc, s in zip(word_count.items(), word_sample)}
# print(word_sample)

data = []
for target_pos in range(CONTEXT_SIZE, len(raw_data) - CONTEXT_SIZE):
    context = []
    for w in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
        if w == 0:
            continue
        context.append(raw_data[target_pos + w])
    data.append((raw_data[target_pos], context))

dataset = PermutedSubsampleedCorpus(data, word_sample)
print(dataset)

dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=1)

model = SkipGramNegativeSample(vocab_size, EMBEDDING_DIM, n_negs=5)
optimizer = optim.SGD(model.parameters(), lr=0.1)

def make_context_vectors(context, word_to_ix):
    context_ixs = [word_to_ix[w] for w in context]
    return torch.tensor(context_ixs, dtype=torch.long)


losses = []
for epoch in range(10):
    total_loss = 0
    for batch_size, (iword, owords) in enumerate(dataloader):

        iword = list(map(lambda x: word_to_ix[x], iword))
        iword = torch.tensor(iword, dtype=torch.long)

        owords = list(map(list, owords))
        owords = np.array(owords).T

        myfunc = np.vectorize(lambda x: word_to_ix[x])
        owords = list(map(myfunc, owords))
        owords = torch.tensor(owords, dtype=torch.long)

        model.zero_grad()
        loss = model(iword, owords)
        loss.backward()
        optimizer.step()

        total_loss += loss
    losses.append(total_loss)
print(losses)
