import argparse
import logging
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm

from speech_datasets import SpeechDataLoader
try:
    from .model import EncoderDecoder
except ImportError:
    from train_example.model import EncoderDecoder


logger = logging.getLogger(__name__)


def parse_args():
    dirname = os.path.dirname(os.path.abspath(__file__))
    default_spm = os.path.join(dirname, "wsj_bpe75.model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--feats_type", type=str, default="fbank",
                        choices=["fbank", "fbank_pitch"])
    parser.add_argument("--precomputed_feats", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=10)

    parser.add_argument("--sentencepiece_model", default=default_spm, type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Configure the logger
    lowest = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', stream=sys.stdout,
                        level=lowest if args.local_rank in [-1, 0] else logging.WARN)

    # Set up Pytorch's distributed backend if needed
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.ddp = lambda model: model
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.ddp = lambda model: torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    # Determine the features to use (pre-computed or not)
    args.transform_conf = os.path.join(dirname, f"{args.feats_type}.yaml")
    if not args.precomputed_feats:
        args.feats_type = "raw"

    return args


def get_data_loader(datasets, args, shuffle, num_workers=7):
    return SpeechDataLoader(
        datasets, shuffle=shuffle, batch_size=args.batch_size,
        spmodel=args.sentencepiece_model,
        transform_conf=args.transform_conf,
        precomputed_feats_type=args.feats_type,
        num_workers=num_workers)


def main():
    args = parse_args()

    logger.info("Initializing data loaders...")
    train = ["librispeech/train-clean-100", "wsj/train_si284"]
    dev = ["librispeech/dev-clean", "wsj/test_dev93"]

    # Wherever possible, please use the SpeechDataLoader as a context manager,
    # as done here. This is because the data loader creates background threads
    # and processes that will not close unless loader.close() is invoked, which
    # is handled automatically by the syntax below.
    #
    # If you are unable to use the `with SpeechDataLoader(...) as loader: ...`
    # syntax, make sure that you call loader.close() manually after you're done
    # using it. If you forget to do this, your program may not terminate!
    #
    with get_data_loader(train, args, True) as train_loader, \
            get_data_loader(dev, args, False) as dev_loader:

        logger.info("Initializing model & optimizer...")
        idim, odim, adim = 80, len(train_loader.tokenizer), 256
        model = EncoderDecoder(n_enc_layers=4, n_dec_layers=2, input_dim=idim,
                               output_dim=odim, attn_dim=adim).to(device=args.device)
        model = args.ddp(model)

        # Set up optimizer & learning rate scheduler (Noam scheduler from
        # "Attention Is All You Need" by Vaswani et al., NeurIPS 2017)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        opt = torch.optim.Adam(model.parameters(), lr=0.02 / math.sqrt(adim))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min((i+1) / args.warmup_steps ** 1.5, 1 / math.sqrt(i+1)))

        # Main training loop
        t0 = time.time()
        for epoch in range(args.num_epochs):
            logger.info(f"Starting epoch {epoch+1}...")
            train_loader.set_epoch(epoch)
            # Each batch is a list of dictionaries (one per utterance)
            for batch in tqdm.tqdm(train_loader):

                # utt_dict["x"] is a FloatTensor of shape (n_frames, feat_dim).
                # This is the sequence of feature vectors extracted for the utterance
                xs = [utt_dict["x"] for utt_dict in batch]
                x_lens = [x.shape[0] for x in xs]
                xs = pad_sequence(xs, batch_first=True).to(device=args.device)

                # utt_dict["labels"] is an IntTensor of shape (n_labels,). This is
                # the sequence of sub-word token indexes obtained by running the
                # SentencePiece model on this utterance's transcript.
                # Note: utt_dict["text"] provides the transcript as a string.
                ys = [utt_dict["labels"] for utt_dict in batch]
                y_lens = [len(y) for y in ys]
                ys = pad_sequence(ys, batch_first=True).to(device=args.device)

                # Our model is a transformer encoder-decoder that we train to
                # minimize the negative log likelihood (cross entropy) loss.
                opt.zero_grad()
                logits = model(xs, x_lens, ys, y_lens)
                loss = loss_fn(logits.view(-1, odim), ys.view(-1))
                loss.backward()
                opt.step()
                sched.step()

                # TODO: log tensorboard metrics during training

            logger.info(f"Evaluating model after epoch {epoch}...")
            # TODO: end of epoch evaluation on dev set

    logger.info(f"[Elapsed]: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
