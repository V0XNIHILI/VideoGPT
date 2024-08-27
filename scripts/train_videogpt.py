import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from videogpt import VideoGPT, VideoData

import torch
torch.set_float32_matmul_precision('high')


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()

    # Add args that originally came from this line below:
    # parser = pl.Trainer.add_argparse_args(parser)
    # ...but don't work anymore in the new version of PyTorch Lightning
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    # parser.add_argument('--amp_level', type=str, default='')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=20*1000)

    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1))

    wandb_logger = WandbLogger(project="moving_mnist_videogpt")

    # from pytorch_lightning.profilers import PyTorchProfiler
    # prof = PyTorchProfiler(dirpath="./", filename="profiler_output.txt", export_to_chrome=True)

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(devices=[0,1,2,3,6,7], accelerator="gpu", strategy="ddp") # gpus=args.gpus
    trainer = pl.Trainer(callbacks=callbacks, max_steps=args.max_steps,
                        #  profiler="simple", max_epochs=1, limit_train_batches=20, limit_val_batches=4,  val_check_interval=20,
                        #  accumulate_grad_batches=3, 
                         logger=wandb_logger, **kwargs)
    trainer.fit(model, data)


if __name__ == '__main__':
    main()

