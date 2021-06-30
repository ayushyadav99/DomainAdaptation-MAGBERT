from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import pickle
import sys
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from bert import MAG_BertForSequenceClassification, BertClassifier, Discriminator
from xlnet import MAG_XLNetForSequenceClassification

from train import pretrain, adapt, evaluate

from argparse_utils import str2bool, seed
from global_configs import MOSEI_ACOUSTIC_DIM, MOSEI_VISUAL_DIM, MOSI_ACOUSTIC_DIM, MOSI_VISUAL_DIM, DEVICE

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=[ "booktube_persuasive", "booktube_expertise", "pom_persuasive", "pom_expertise"], default="longerFeat_pom_persuasive_!sp")
parser.add_argument("--tgt_dataset", type=str,
                    choices=[ "booktube_persuasive", "booktube_expertise", "pom_persuasive", "pom_expertise"], default="booktube_persuasive_big")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
# parser.add_argument("--n_epochs", type=int, default=2) 
# default = 40
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased", "xlnet-base-cased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
# ----------------------------------------------------------
parser.add_argument('--pretrain', default=True, action='store_true', help='Force to pretrain source encoder/classifier')
parser.add_argument('--adapt', default=True, action='store_true', help='Force to adapt target encoder')
parser.add_argument('--seed', type=int, default=42, help="Specify random state")
parser.add_argument('--train_seed', type=int, default=42, help="Specify random state")
parser.add_argument('--load', default=False, action='store_true', help="Load saved model")
parser.add_argument('--alpha', type=float, default=0.8, help="Specify adversarial weight")
parser.add_argument('--beta', type=float, default=0.5, help="Specify KD loss weight")
parser.add_argument('--temperature', type=int, default=20, help="Specify temperature")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument('--pre_epochs', type=int, default=50, help="Specify the number of epochs for pretrain")
parser.add_argument('--pre_log_step', type=int, default=1, help="Specify log step size for pretrain")
parser.add_argument('--num_epochs', type=int, default=100, help="Specify the number of epochs for adaptation")
parser.add_argument('--log_step', type=int, default=5, help="Specify log step size for adaptation")

args = parser.parse_args()


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob, dataset):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        self.dataset = dataset


def convert_to_features(dataset, examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example
        # print(acoustic)
        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []
        visual = np.array(visual)
        acoustic = np.array(acoustic)
        #print(len(visual)) 
        #print(len(visual[0]))
        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)
        # print(acoustic.shape)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        if args.model == "bert-base-uncased":
            prepare_input = prepare_bert_input
        elif args.model == "xlnet-base-cased":
            prepare_input = prepare_xlnet_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            dataset, tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_bert_input(dataset, tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    if dataset == "mosi_bin": 
        ACOUSTIC_DIM = MOSI_ACOUSTIC_DIM
        VISUAL_DIM = MOSI_VISUAL_DIM
    else:
        ACOUSTIC_DIM = 37 
        VISUAL_DIM = 328
        # VISUAL_DIM = 31
        #ACOUSTIC_DIM = 43
        #VISUAL_DIM = 711
     

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    # print(acoustic_zero.shape, acoustic.shape)
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def prepare_xlnet_input(dataset, tokens, visual, acoustic, tokenizer):
    
    if dataset == "mosi":
        ACOUSTIC_DIM = MOSI_ACOUSTIC_DIM
        VISUAL_DIM = MOSI_VISUAL_DIM
    else:
        ACOUSTIC_DIM = MOSEI_ACOUSTIC_DIM
        VISUAL_DIM = MOSEI_VISUAL_DIM
    
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    PAD_ID = tokenizer.pad_token_id

    # PAD special tokens
    tokens = tokens + [SEP] + [CLS]
    audio_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, audio_zero, audio_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(tokens) - 1) + [2]

    pad_length = (args.max_seq_length - len(segment_ids))

    # then zero pad the visual and acoustic
    audio_padding = np.zeros(pad_length, ACOUSTIC_DIM)
    acoustic = np.concatenate((audio_padding, acoustic))

    video_padding = np.zeros(pad_length, VISUAL_DIM)
    visual = np.concatenate((video_padding, visual))

    input_ids = [PAD_ID] * pad_length + input_ids
    input_mask = [0] * pad_length + input_mask
    segment_ids = [3] * pad_length + segment_ids

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    if model == "bert-base-uncased":
        return BertTokenizer.from_pretrained(model)
    elif model == "xlnet-base-cased":
        return XLNetTokenizer.from_pretrained(model)
    else:
        raise ValueError(
            "Expected 'bert-base-uncased' or 'xlnet-base-cased, but received {}".format(
                model
            )
        )


def get_appropriate_dataset(dataset, data):

    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(dataset, data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle, encoding="utf-8")

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(args.dataset, train_data)
    dev_dataset = get_appropriate_dataset(args.dataset, dev_data)
    test_dataset = get_appropriate_dataset(args.dataset, test_data)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        # * args.n_epochs
        * args.pre_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    with open(f"datasets/{args.tgt_dataset}.pkl", "rb") as tgt_handle:
        tgt_data = pickle.load(tgt_handle)

    tgt_train_data = tgt_data["train"]
    tgt_dev_data = tgt_data["dev"]
    tgt_test_data = tgt_data["test"]

    tgt_train_dataset = get_appropriate_dataset(args.tgt_dataset, tgt_train_data)
    tgt_dev_dataset = get_appropriate_dataset(args.tgt_dataset, tgt_dev_data)
    tgt_test_dataset = get_appropriate_dataset(args.tgt_dataset, tgt_test_data)

    num_train_optimization_steps = (
        int(
            len(tgt_train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        # * args.n_epochs
        * args.pre_epochs
    )

    tgt_train_dataloader = DataLoader(
        tgt_train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    tgt_dev_dataloader = DataLoader(
        tgt_dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    tgt_test_dataloader = DataLoader(
        tgt_test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
        tgt_train_dataloader,
        tgt_dev_dataloader,
        tgt_test_dataloader,  
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps: int):
    src_multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob,
        dataset=args.dataset
    )

    tgt_multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob,
        dataset=args.tgt_dataset
    )
    if args.model == "bert-base-uncased":
        model = MAG_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=src_multimodal_config, num_labels=1,
        )
        tgt_model = MAG_BertForSequenceClassification.from_pretrained(
            args.model, multimodal_config=tgt_multimodal_config, num_labels=1,
        ) 
    elif args.model == "xlnet-base-cased":
        model = MAG_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=src_multimodal_config, num_labels=1
        )
        tgt_model = MAG_XLNetForSequenceClassification.from_pretrained(
            args.model, multimodal_config=tgt_multimodal_config, num_labels=1
        )

    model.to(DEVICE)
    tgt_model.to(DEVICE)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, tgt_model, optimizer, scheduler

def main():
    wandb.init(project="MAG")
    wandb.config.update(args)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
        tgt_train_data_loader,
        tgt_dev_data_loader,
        tgt_test_data_loader,
    ) = set_up_data_loader()

    src_classifier = BertClassifier().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    model, tgt_model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    """ Adding pretrain step  """
    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain: 
        src_encoder, src_classifier = pretrain(
            args, model, src_classifier, train_data_loader, optimizer, scheduler)
    
    # eval source model
    print("=== Evaluating classifier for source domain ===")

    print("=== Src Enc and Cls on SrcTrainData ===")
    evaluate(args, src_encoder, src_classifier, train_data_loader)
    print("=== Src Enc and Cls on SrcTestData ===")
    evaluate(args, src_encoder, src_classifier, test_data_loader)
    print("=== Src Enc and Cls on TgtTrainData ===")
    evaluate(args, src_encoder, src_classifier, tgt_train_data_loader)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if args.adapt:
        print("=== Adapt called ===")
        tgt_model.load_state_dict(model.state_dict())
        tgt_model = adapt(args, model, tgt_model, discriminator,
                            src_classifier, train_data_loader, tgt_train_data_loader, tgt_test_data_loader)

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    evaluate(args, model, src_classifier, tgt_test_data_loader)
    print(">>> domain adaption <<<")
    evaluate(args, tgt_model, src_classifier, tgt_test_data_loader)

if __name__ == "__main__":
    main()