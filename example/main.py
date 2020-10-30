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

from speech_datasets import SpeechDataLoader as SDL
from example.model import EncoderDecoder
from example.utils import edit_dist


logger = logging.getLogger(__name__)
dirname = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    default_spm = os.path.join(dirname, "resources", "librispeech_bpe2000.model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--feats_type", type=str, default="fbank",
                        choices=["fbank", "fbank_pitch"],
                        help="The type of features you would like to use.")
    parser.add_argument("--precomputed_feats", action="store_true", default=False,
                        help="Specify if you want to use pre-computed features, "
                             "instead of computing them online.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for the data loader to compute "
                             "feature transformations.")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--max_len", type=int, default=480,
                        help="Maximum effective utterance length for the "
                             "purposes of computing batch size.")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="Number of steps to warm up the learning rate.")
    parser.add_argument("--num_epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint to load.")

    parser.add_argument("--sentencepiece_model", default=default_spm, type=str,
                        help="Sentencepiece model to use for tokenization.")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Specify if you want debug-level logs.")
    args = parser.parse_args()

    # Configure the logger
    if args.debug:
        level = logging.DEBUG
    elif args.local_rank in [-1, 0]:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(format=f'%(asctime)s - %(levelname)s - %(name)s - rank={args.local_rank} - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', stream=sys.stdout, level=level)

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

    return args


def get_data_loader(datasets, args, shuffle, train):
    transform_conf = os.path.join(dirname, "resources", f"{args.feats_type}.yaml")
    feats_type = "raw" if args.precomputed_feats else args.feats_type
    return SDL(
        datasets, shuffle=shuffle, train=train,
        batch_size=args.batch_size, max_len=args.max_len,
        spmodel=args.sentencepiece_model,
        transform_conf=transform_conf,
        precomputed_feats_type=feats_type,
        num_workers=args.num_workers)


def main():
    args = parse_args()

    logger.info("Initializing data loaders...")
    train = ["librispeech/train-clean-100", "librispeech/train-clean-360"]
    dev = ["librispeech/dev-clean"]

    # Wherever possible, please use the SpeechDataLoader as a context manager,
    # as done here. This is because the data loader creates background threads
    # and processes that will not close unless loader.close() is invoked, which
    # is handled automatically by the syntax below.
    #
    # If you are unable to use the `with SpeechDataLoader(...) as loader: ...`
    # syntax, make sure that you call loader.close() manually after you're done
    # using it. If you forget to do this, your program may not terminate!
    #
    with get_data_loader(train, args, shuffle=True, train=True) as train_loader, \
            get_data_loader(dev, args, shuffle=False, train=False) as dev_loader:

        # Initialize dimensions & model
        logger.info("Initializing model & optimizer...")
        idim = 80 if args.feats_type == "fbank" else 83
        odim, adim = len(train_loader.tokenizer), 256
        model = EncoderDecoder(n_enc_layers=8, n_dec_layers=4, input_dim=idim,
                               output_dim=odim, attn_dim=adim).to(device=args.device)

        # Load a model checkpoint if desired
        if args.checkpoint is not None:
            state_dict = torch.load(args.checkpoint, map_location=args.device)
            model.load_state_dict(state_dict["model"])
            best_ter, first_epoch = state_dict["best_ter"], state_dict["epoch"] + 1
        else:
            best_ter, first_epoch = 1e20, 1

        # Set up Distributed Data Parallel if desired
        model = args.ddp(model)

        # Set up optimizer & learning rate scheduler (Noam scheduler from
        # "Attention Is All You Need" by Vaswani et al., NeurIPS 2017)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        opt = torch.optim.Adam(model.parameters(), lr=0.02 / math.sqrt(adim))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min((i+1) / args.warmup_steps ** 1.5, 1 / math.sqrt(i+1)))

        # Main training loop
        t0 = time.time()
        sos_eos = torch.tensor(train_loader.tokenizer.tokens2ids(["<sos/eos>"]))
        for epoch in range(first_epoch, args.num_epochs + 1):
            logger.info(f"Starting epoch {epoch}...")
            train_loader.set_epoch(epoch)
            # Each batch is a list of dictionaries (one per utterance)
            for batch in tqdm.tqdm(train_loader, disable=args.local_rank not in [-1, 0]):

                # utt_dict["x"] is a FloatTensor of shape (n_frames, feat_dim).
                # This is the sequence of feature vectors extracted for the utterance
                xs = [utt_dict["x"] for utt_dict in batch]
                x_lens = [x.shape[0] for x in xs]
                xs = pad_sequence(xs, batch_first=True).to(device=args.device)

                # utt_dict["labels"] is an IntTensor of shape (n_labels,). This is
                # the sequence of sub-word token indexes obtained by running the
                # SentencePiece model on this utterance's transcript.
                # Note: utt_dict["text"] provides the transcript as a string.
                ys = [torch.cat((sos_eos, utt_dict["labels"], sos_eos))
                      for utt_dict in batch]
                y_lens = [len(y) for y in ys]
                ys = pad_sequence(ys, batch_first=True).to(device=args.device)

                # Our model is a transformer encoder-decoder that we train to
                # minimize the negative log likelihood (cross entropy) loss.
                opt.zero_grad()
                logits = model(xs, x_lens, ys, y_lens)[:, :-1]
                loss = loss_fn(logits.reshape(-1, odim), ys[:, 1:].reshape(-1))
                loss.backward()
                opt.step()
                sched.step()

            logger.info(f"Evaluating model after {epoch} epochs...")
            n_char, n_edits = 0, 0
            for dev_batch in tqdm.tqdm(dev_loader, disable=args.local_rank not in [-1, 0]):
                # Get inputs as above
                xs = [utt_dict["x"] for utt_dict in dev_batch]
                x_lens = [x.shape[0] for x in xs]
                xs = pad_sequence(xs, batch_first=True).to(device=args.device)

                # Get ground truth outputs as above
                ys = [torch.cat((sos_eos, utt_dict["labels"], sos_eos))
                      for utt_dict in dev_batch]
                y_lens = [len(y) for y in ys]
                ys = pad_sequence(ys, batch_first=True).to(device=args.device)

                # Model output is a padded version of the y's.
                with torch.no_grad():
                    yhats = model(xs, x_lens, ys, y_lens).argmax(dim=-1)
                    n_char += sum(y_lens) - len(y_lens)
                    n_edits += sum(edit_dist(yhat[0:n-1].tolist(), y[1:n].tolist())
                                   for yhat, y, n in zip(yhats, ys, y_lens))

            # Log the token error rate of the model
            if args.local_rank != -1:
                n_edits_char = torch.tensor([n_edits, n_char], dtype=torch.long, device=args.device)
                dist.all_reduce(n_edits_char, op=dist.ReduceOp.SUM)
                n_edits, n_char = n_edits_char.tolist()
            ter = n_edits / n_char
            logger.info(f"Token error rate after {epoch} epochs: {ter:.4f}\n")

            # Save the checkpoint
            if args.local_rank in [-1, 0]:
                model_to_save = model.module if hasattr(model, "module") else model
                state = {"model": model_to_save.state_dict(),
                         "epoch": epoch, "ter": ter, "best_ter": min(ter, best_ter)}
                os.makedirs("checkpoint", exist_ok=True)
                torch.save(state, f"checkpoint/{str(epoch).zfill(3)}.pt")
                if ter < best_ter:
                    best_ter = ter
                    torch.save(state, f"checkpoint/best.pt")

    logger.info(f"[Elapsed]: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
