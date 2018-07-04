import torch
import numpy as np
from model import *
from utils import *
from torch.autograd import Variable

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = data
        tgt = data
        yield Batch(src, tgt, 0)

test = torch.randn(4,1)
print(test.size())

tmp_1 = [[torch.rand(2,3)],[torch.rand(2,3)]]
tmp_2 = (torch.rand(2,3),torch.rand(2,3))
res_1 = [[o[0][:,1:2]] for o in tmp_1]
res_2 = [[o[:,1:2]] for o in tmp_2]
print(res_1)
print(res_2)

V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)


model = build_model(src_vocab=V, tgt_vocab=V, N=2)


model_opt = NoamOpt(model_size=model.src_embed[0].d_model, factor=1, warmup=400,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))




for epoch in range(10):
    model.train()
    run_epoch(data_iter=data_gen(V, 2, 20), model=model,
              loss_compute=SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))


model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))




