import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import config

class TClass(nn.Module):

    def __init__(self,class_count,emb_dims,lstm_size=30):
        super(TClass, self).__init__()
        e_num,e_dim=emb_dims
        self.embedding=nn.Embedding(num_embeddings=e_num,embedding_dim=e_dim,padding_idx=0,sparse=True)
        #self.embedding.weight.data.fill_(0.01)
        self.bilstm1=nn.LSTM(input_size=self.embedding.embedding_dim,hidden_size=lstm_size,num_layers=1,bidirectional=True)
        self.dense1=nn.Linear(in_features=self.bilstm1.hidden_size*self.bilstm1.num_layers*2,out_features=class_count)
        #self.dense1=nn.Linear(in_features=self.bilstm1.hidden_size,out_features=class_count)

    def forward(self,minibatch):
        #print("minibatch",minibatch)
        minibatch_emb=self.embedding(minibatch)
        #print("mbemb",minibatch_emb)
        bilstm_out,(h_n,c_n)=self.bilstm1(minibatch_emb)
        #print("bilstm_out-1",bilstm_out[-1])
        #print("h_n",h_n.size())
        layers_dirs,batch,feats=h_n.size()
        #print("h_n",h_n)
        #print("hn0",h_n[0])
        steps,batch,feats=bilstm_out.size()
        #h_n_linin=bilstm_out[steps-1,:,:]
        h_n_linin=h_n.transpose(0,1).contiguous().view(batch,-1)#.contiguous().view(batch,-1)
        #print("h_n_linin",h_n_linin)
        #print("\n\n\n\n")
        dense_out=F.tanh(self.dense1(h_n_linin))
        #print("dense_out",dense_out)
        return F.softmax(dense_out,dim=1)
        
        
        
    #     # 1 input image channel, 6 output channels, 5x5 square convolution
    #     # kernel
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     # an affine operation: y = Wx + b
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     # Max pooling over a (2, 2) window
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #     # If the size is a square you can only specify a single number
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

if __name__=="__main__":
    x=TClass(class_count=2)
