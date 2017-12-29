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

import sklearn.feature_extraction
import time


def minibatched_3dim(data,batch_size):
    seq_count,word_count,char_count=data.size()
    seq_count_mbatch_aligned=(seq_count//batch_size)*batch_size
    data_batched=data[:seq_count_mbatch_aligned].transpose(0,1).contiguous().view(word_count,seq_count//batch_size,-1,char_count)
    return data_batched


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cpu', dest='cuda', default=True, action="store_false", help='Force CPU. Default is to use CUDA/GPU.')
    parser.add_argument('--batch-size',default=100, type=int, help='Batch size. Default %(default)d')
    args = parser.parse_args()

    tokenizer_func=sklearn.feature_extraction.text.CountVectorizer().build_tokenizer()
    
    config.set_cuda(args.cuda)
    
    torch.manual_seed(1)
    config.torch_mod.manual_seed(1) #cuda seed, if enabled

    start=time.clock()
    texts,classes=data_imdb.read_data("train",5000) #X is list of texts
    texts=[tokenizer_func(t.lower()) for t in texts]
    t2i_txt=t2i.T2I()
    t2i_chr=t2i.T2I()
    t2i_cls=t2i.T2I(with_padding=None,with_unknown=None)
    print("Texts loaded at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    X_wrd=t2i_txt(texts) #[[word01,word02,...],[word11,word12...]] (as integers)
    X_chr=t2i_chr(texts,string_as_sequence=True) #[[[w,o,r,d,0,1],[w,o,r,d,0,2],..],[[w,o,r,d,1,1],[w,o,r,d,1,2],..]] (as integers)
    print("T2I completed at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    X_wrd_t=t2i.to_torch_long_tensor(X_wrd,[300]) #max 300 words per text
    X_chr_t=t2i.to_torch_long_tensor(X_chr,[300,10]) #max 300 words per text, max 20 characters per word
    print("Torch tensors done at {:.1f}sec".format(time.clock()-start),file=sys.stderr)
    #Now X_wrd_t is (seq X word)  and X_chr_t is (seq X word X char)
    #
    # we need word x minibatch x seq  and   öööö I'll do this later
    X_wrd_t_batched=t2i.torch_minibatched_2dim(X_wrd_t,args.batch_size)
    _,batches,_=X_wrd_t_batched.size()
    del X_wrd_t
    
    X_chr_t_batched=minibatched_3dim(X_chr_t,args.batch_size)
    del X_chr_t
    
    Y=t2i_cls(classes) #class indices
    Y_t_batched=torch.LongTensor(Y)[:batches*args.batch_size].view(batches,-1).contiguous() #minibatched
    
    network=model.TClass(len(t2i_cls.idict),wrd_emb_dims=(len(t2i_txt.idict),50),chr_emb_dims=(len(t2i_chr.idict),50))
    if config.cuda:
        network.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.2, momentum=0.9)
    
    batches=X_wrd_t_batched.size(1)
    print("X_wrd_t_batched size",X_wrd_t_batched.size())
    print("X_chr_t_batched size",X_chr_t_batched.size())
    print("Y_t_batched size",Y_t_batched.size())
    while True:
        accum_loss=0
        accum_acc=0
        for batch_idx in range(batches):
            #print("Batch",batch_idx)
            minibatch_wrd_t=X_wrd_t_batched[:,batch_idx,:]
            minibatch_chr_t=X_chr_t_batched[:,batch_idx,:,:]
            #print("minibatch_t-size",minibatch_t.size())
            minibatch_wrd_tv=Variable(minibatch_wrd_t)
            minibatch_chr_tv=Variable(minibatch_chr_t)
            gold_classes_tv=Variable(Y_t_batched[batch_idx,:])
            if config.cuda:
                minibatch_wrd_tv=minibatch_wrd_tv.cuda()
                minibatch_chr_tv=minibatch_chr_tv.cuda()
                gold_classes_tv=gold_classes_tv.cuda()
                
            optimizer.zero_grad()
            outputs=network(minibatch_wrd_tv,minibatch_chr_tv)
            #print("outputs",outputs)
            #print("gold",gold_classes_tv)
            values,indices=outputs.max(1)
            accum_acc+=float(torch.sum(indices.eq(gold_classes_tv)))/minibatch_wrd_t.size(1)
            loss=criterion(outputs,gold_classes_tv)
            print("minibatch loss",float(loss))
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
        

