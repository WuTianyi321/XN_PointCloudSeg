import sys
import os
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,path)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from data.dataset import XN_PointCloudDataset
from model.pointnet import PointNetDenseCls
import torch.utils.data
from pytorch_lightning.metrics import functional as FM

class ClassificationTask(pl.LightningModule):
     def __init__(self, model):
         super().__init__()
         self.model = model

     def training_step(self, batch, batch_idx):
         x, y = batch
         #x=x.transpose(2, 1)
         y_hat, trans, trans_feat = self.model(x.transpose(2, 1))
         y_hat = y_hat.view(-1, 2)
         #loss = F.cross_entropy(y_hat, y)
         #loss = F.nll_loss(y_hat, y.view(-1))
         loss = F.nll_loss(y_hat, y.view(-1),weight =torch.cuda.FloatTensor([2,8]))
         acc = FM.accuracy(torch.exp(y_hat.view(-1,2)), y.view(-1))
         #self.log('training_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
         return loss

     def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, trans, trans_feat = self.model(x.transpose(2, 1))
        y_hat = y_hat.view(-1, 2)
        #y_hat = y_hat.view(-1, 1)[:, 0]
        #loss = F.cross_entropy(y_hat, y)
        #loss = F.nll_loss(y_hat, y.view(-1))
        loss = F.nll_loss(y_hat, y.view(-1),weight =torch.cuda.FloatTensor([2,8]))
        acc = FM.accuracy(torch.exp(y_hat.view(-1,2)), y.view(-1))
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        self.log('val_acc', metrics['val_acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics

     def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        #self.log('val_acc', metrics['val_acc'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(metrics)


     def configure_optimizers(self):
         return torch.optim.Adam(self.model.parameters(), lr=0.0005)

npoints=5000
# data
root='D:/Users/WuTianyi/OneDrive - wutianyidev/NEU/课程/MATH7243 Machine Learning/大作业/Cloud Sample Data'
dataset = XN_PointCloudDataset(
    root=root
    ,npoints=npoints)

#mnist_train, mnist_val = random_split(dataset, [55000, 5000])

#train_loader = DataLoader(mnist_train, batch_size=32)
#val_loader = DataLoader(mnist_val, batch_size=32)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    #num_workers=int(opt.workers)
    )

test_dataset = XN_PointCloudDataset(
    root=root
    ,split='test'
    ,npoints=npoints)
val_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    #num_workers=int(opt.workers)
    )
# model

model = ClassificationTask(PointNetDenseCls(k=2))

# training
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)