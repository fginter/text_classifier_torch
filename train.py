import argparse
import config
import data_imdb
import t2i
import model

import torch
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim
import sys



def data2padded(X,batch_size,max_seq_len=0):
    example_count=len(X)
    seq_len=max(len(ex) for ex in X)
    seq_len=min(seq_len,max_seq_len)
    example_count=(example_count//batch_size)*batch_size #trim to batch size
    X=X[:example_count]
    data=torch.LongTensor(seq_len,example_count).zero_()
    for e_idx,example in enumerate(X):
        for t_idx,t in enumerate(example[:seq_len]):
            data[t_idx,e_idx]=t
    data=data.view((seq_len,example_count//batch_size,-1)) #subdivide to minibatches
    return data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Force CPU. Default is to use CUDA/GPU.')
    parser.add_argument('--batch-size',default=100, type=int, help='Batch size. Default %(default)d')
    args = parser.parse_args()
    
    config.set_cuda(args.cuda)
    
    torch.manual_seed(1)
    config.torch_mod.manual_seed(1) #cuda seed, if enabled

    
    texts,classes=data_imdb.read_data("train") #X is list of texts
    texts=[t.lower().split() for t in texts]
    t2i_txt=t2i.T2I()
    t2i_cls=t2i.T2I(with_padding=None,with_unknown=None)

    X=t2i_txt(texts)#,string_as_sequence=True) #character embeddings
    Y=t2i_cls(classes) #class indices
    X_t=data2padded(X,args.batch_size,max_seq_len=300)
    Y_t=torch.LongTensor(Y).view(len(Y)//args.batch_size,-1) #minibatched
    
    network=model.TClass(len(t2i_cls.idict),emb_dims=(len(t2i_txt.idict),50))
    if config.cuda:
        network.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.2, momentum=0.9)
    
    batches=X_t.size(1)
    print("X_t size",X_t.size())
    while True:
        accum_loss=0
        accum_acc=0
        for batch_idx in range(batches):
            #print("Batch",batch_idx)
            minibatch_t=X_t[:,batch_idx,:]
            #print("minibatch_t-size",minibatch_t.size())
            minibatch_tv=Variable(minibatch_t)
            gold_classes_tv=Variable(Y_t[batch_idx,:])
            if config.cuda:
                minibatch_tv=minibatch_tv.cuda()
                gold_classes_tv=gold_classes_tv.cuda()
                
            optimizer.zero_grad()
            outputs=network(minibatch_tv)
            #print("outputs",outputs)
            #print("gold",gold_classes_tv)
            values,indices=outputs.max(1)
            accum_acc+=float(torch.sum(indices.eq(gold_classes_tv)))/minibatch_t.size(1)
            loss=criterion(outputs,gold_classes_tv)
            accum_loss+=float(loss)
            loss.backward()
            #print("STATEDICT",list(network.state_dict().keys()))
            #print("EMBGRAD",dict(network.named_parameters())["embedding.weight"].grad[:30])
            #print("DENSEGRAD",dict(network.named_parameters())["dense1.weight"].grad)
            #for p in network.parameters():
            #    print(p.grad)
            #print("LOSS",loss)
            #print()
            optimizer.step()
            #break
            #print("linout",network(minibatch_t))
        print("loss",accum_loss/batches)
        print("train-acc",accum_acc/batches*100)
        print()
        

