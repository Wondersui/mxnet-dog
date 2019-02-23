import mxnet as mx
import d2lzh as d2l
import math
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import data as gdata,loss as gloss,model_zoo,nn
import os
import shutil
import time
import numpy as np
import collections
def reorg_train_valid(data_dir,train_dir,input_dir,valid_ratio,index_label):
    # print(collections.Counter(index_label.values()))
    min_n_train_per_label = (
        collections.Counter(index_label.values()).most_common()[:-2:-1][0][1])
    n_valid_per_label = math.floor(min_n_train_per_label*valid_ratio)
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir,train_dir)):
        idx = train_file.split('.')[0]
        label = index_label[idx]
        d2l.mkdir_if_not_exist([data_dir,input_dir,'train_valid',label])
        shutil.copy(os.path.join(data_dir,train_dir,train_file),
                    os.path.join(data_dir,input_dir,'train_valid',label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            d2l.mkdir_if_not_exist([data_dir,input_dir,'valid',label])
            shutil.copy(os.path.join(data_dir,train_dir,train_file),
                        os.path.join(data_dir,input_dir,'valid',label))
            label_count[label] = label_count.get(label,0) + 1
        else:
            d2l.mkdir_if_not_exist([data_dir,input_dir,'train',label])
            shutil.copy(os.path.join(data_dir,train_dir,train_file),
                        os.path.join(data_dir,input_dir,'train',label))

def reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio):
    with open(os.path.join(data_dir,label_file),'r') as f:
        lines = f.readlines()[1:]
        tokens = [l.strip().split(',') for l in lines]
        # print(tokens[:10])
        idx_label = dict((idx,label) for idx,label in tokens)
        # print(idx_label)
    reorg_train_valid(data_dir,train_dir,input_dir,valid_ratio,idx_label)
    d2l.mkdir_if_not_exist([data_dir,input_dir,'test','unknown'])
    for test_file in os.listdir(os.path.join(data_dir,test_dir)):
        shutil.copy(os.path.join(data_dir,test_dir,test_file),
                    os.path.join(data_dir,input_dir,'test','unknown'))

reorg_data = False
data_dir = '/home/suizhehao/mxnet-cv-project/dog_data'
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'
input_dir, valid_ratio = 'train_valid_test', 0.1
batch_size = 64
if reorg_data:
    reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio)

transform_train = gdata.vision.transforms.Compose([
    gdata.vision.transforms.RandomResizedCrop(224,scale=(0.08,1.0),
                                              ratio=(3.0/4.0,4.0/3.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    gdata.vision.transforms.RandomLighting(0.1),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485,0.456,0.406],
                                      [0.229,0.224,0.225])
])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.CenterCrop(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])

train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir,input_dir,'train'),flag=1)
valid_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir,input_dir,'valid'),flag=1)
train_valid_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'train_valid'), flag=1)
test_ds = gdata.vision.ImageFolderDataset(
    os.path.join(data_dir, input_dir, 'test'), flag=1)

train_iter = gdata.DataLoader(train_ds.transform_first(transform_train),batch_size=batch_size,
                              shuffle=True,last_batch='keep',num_workers=8)
valid_iter = gdata.DataLoader(valid_ds.transform_first(transform_test),
                              batch_size, shuffle=True, last_batch='keep',num_workers=8)
train_valid_iter = gdata.DataLoader(train_valid_ds.transform_first(
    transform_train), batch_size, shuffle=True, last_batch='keep',num_workers=8)
test_iter = gdata.DataLoader(test_ds.transform_first(transform_test),
                             batch_size, shuffle=False, last_batch='keep',num_workers=8)

def classifier():
    net = nn.HybridSequential()
    net.add(nn.Dense(256,activation='relu'))
    net.add(nn.Dense(120))
    return net
ctx = d2l.try_gpu()
net = nn.HybridSequential()
finetune_net = model_zoo.vision.resnet101_v2(pretrained=True,ctx=ctx)
features = finetune_net.features
for _,w in features.collect_params().items():
    w.grad_req = 'null'

with net.name_scope():
    net.add(features)
    net.add(classifier())
    net[1].initialize(init=init.Xavier(),ctx=ctx)
# print(net)
# net.hybridize()
# def get_net(ctx):
#     finetune_net = model_zoo.vision.resnet101_v2(pretrained=True)
#     # feature_net = finetune_net.features
#     # with
#     # new_net = nn.HybridSequential(prefix='')
#     # new_net.add(feature_net)
#     # new_net.add(nn.Dense(256,activation='relu'))
#     # new_net.add(nn.Dense(120))
#     # # finetune_net.output_new.initialize(init=init.Xavier(),ctx=ctx)
#     # new_net.collect_params().reset_ctx(ctx)
#     return finetune_net
#

#
# net = get_net(ctx)
# print(net)


loss = gloss.SoftmaxCrossEntropyLoss()

def accuracy(output,label):
    preds = nd.argmax(output,axis=1)
    pred_label = preds.asnumpy().astype('int32')
    label = label.asnumpy().astype('int32')
    label = label.flat
    pred_label = pred_label.flat
    sum = (pred_label == label).sum()
    return sum


acc = mx.metric.Accuracy()

def evaluate_loss(data_iter,net,ctx):
    l_sum,n,acc = 0.0,0.0,0.0
    for x,y in data_iter:
        y = y.as_in_context(ctx)
        outputs = net(x.as_in_context(ctx))
        l_sum += loss(outputs,y).sum().asscalar()
        acc+= accuracy(outputs,y)
        n += y.size
    return l_sum / n,acc / n

def train(net,train_iter,valid_iter,num_epochs,lr,wd,ctx,lr_period,lr_decay):
    trainer = gluon.Trainer(net.collect_params(),'sgd',
                            {'learning_rate':lr,'momentum':0.9,'wd':wd})
    for epoch in range(num_epochs):
        train_l_sum,n,start = 0.0,0,time.time()
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for x,y in train_iter:
            X,Y = x.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                outputs = net(X)
                l = loss(outputs,Y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.asscalar()
            train_acc += accuracy(outputs,Y)
            n += y.size
        time_s = "time %.2f sec"%(time.time()-start)
        if valid_iter is not None:
            valid_loss,valid_acc = evaluate_loss(valid_iter,net,ctx)
            epoch_s = ("epoch %d, train loss %f, valid loss %f,train acc %f,valid acc %f "
                       % (epoch + 1, train_l_sum / n, valid_loss,train_acc/n,valid_acc))
        else:
            epoch_s = ("epoch %d, train loss %f, "
                       % (epoch + 1, train_l_sum / n))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
        if (epoch+1) % 6 == 0:
            net.save_parameters('resnet-101v2-%d'%epoch)

num_epochs, lr, wd =  24, 0.1, 1e-4
lr_period, lr_decay = 6, 0.1
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)