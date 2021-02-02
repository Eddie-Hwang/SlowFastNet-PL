import argparse

import pytorch_lightning as pl
import sh
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import UCF101_Data
from model import FastSlowNet


def get_args():
    parser = argparse.ArgumentParser()

    # dataset related
    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--frame_sample_rate', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data', type=str, default='./dataset') #change dir name

    # trainer related
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--log', type=str, default='./log')

    # model related
    parser.add_argument('--class_num', type=int, default=101)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=10)
    
    return parser.parse_args()

def get_callbacks(dirpath):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=dirpath,
        filename='sf-trained-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    early_stopping = EarlyStopping(monitor='val_loss')

    return [checkpoint_callback, early_stopping]


def main(args):
    print(args)
    sh.rm('-r', '-f', args.log)

    model = FastSlowNet(
        class_num=args.class_num,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        step_size=args.step_size,
    )   
    
    ucf101 = UCF101_Data(
        data=args.data,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = pl.Trainer(
        default_root_dir=args.log,
        gpus=args.gpus,
        max_epochs=args.epochs,
        fast_dev_run=args.debug,
        callbacks=get_callbacks(args.log),
    )
    trainer.fit(model, ucf101)


if __name__=='__main__':
    args = get_args()
    main(args)
