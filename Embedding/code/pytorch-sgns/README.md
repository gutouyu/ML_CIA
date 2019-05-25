# PyTorch SGNS

Word2Vec's **SkipGramNegativeSampling** in Python.

Yet another but quite general [negative sampling loss](https://arxiv.org/abs/1310.4546) implemented in [PyTorch](http://www.pytorch.org).

It can be used with ANY embedding scheme! Pretty fast, I bet.

```python
vocab_size = 20000
word2vec = Word2Vec(vocab_size=vocab_size, embedding_size=300)
sgns = SGNS(embedding=word2vec, vocab_size=vocab_size, n_negs=20)
optim = Adam(sgns.parameters())
for batch, (iword, owords) in enumerate(dataloader):
    loss = sgns(iword, owords)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

New: support negative sampling based on word frequency distribution (0.75th power) and subsampling (resolving word frequency imbalance).

To test this repo, place a space-delimited corpus as `data/corpus.txt` then run `python preprocess.py` and `python train.py --weights --cuda` (use `-h` option for help).
