import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from dataset import VideoDataset
from slowfastnet import Bottleneck, SlowFast


class FastSlowNet(pl.LightningModule):
    def __init__(self, class_num, lr, weight_decay, momentum, step_size):
        super().__init__()
        self.save_hyperparameters()
        
        self.slowfast = SlowFast(Bottleneck, [3, 4, 6, 3]) # resnet 50
        self.fc = nn.Linear(self.slowfast.fast_inplanes+2048, class_num, bias=False)

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step_size = step_size

        self.criterion = nn.CrossEntropyLoss()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def forward(self, x):
        x = self.slowfast(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]
        outputs = self.forward(inputs)
        
        loss = self.criterion(outputs, labels)
        prec1, prec5 = self.accuracy(
            output=outputs.data,
            target=labels,
            topk=(1,5),
        )
        acc = {'Top-1': prec1.item(), 'Top-5': prec5.item()}
        
        # logging
        self.log('tr_loss', loss, prog_bar=False, logger=True)
        self.log('tr_accuracy', acc, prog_bar=False, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        labels = batch[1]
        outputs = self.forward(inputs)

        loss = self.criterion(outputs, labels)
        prec1, prec5 = self.accuracy(
            output=outputs.data,
            target=labels,
            topk=(1,5),
        )
        acc = {'Top-1': prec1.item(), 'Top-5': prec5.item()}
        
        # logging
        self.log('val_loss', loss, prog_bar=False, logger=True)
        self.log('val_accuracy', acc, prog_bar=False, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum
        )
        # scheduler = lr_scheduler.LambdaLR(
        #     optimizer,
        #     lambda epoch: 0.1 ** (epoch // 30)
        # )
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.step_size
        )
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min'
        # )
        return [optimizer], [scheduler]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler,
        #     'monitor': 'val_loss'
        # }
