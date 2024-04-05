import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR


# Dataset Config
from configs import config
from configs import parse_args
from dataset import mtg_cards
from dataset import mtg_decks

from distributed_backends import distributed_utils
from transformers import AutoModelForCausalLM



def main():
    
    args = parse_args()
    
    # distr_backend = distributed_utils.set_backend_from_args(config)
    # distr_backend.initialize()

    ds = mtg_decks(**config.deckDataset, cards=config.cardDataset)

    # if distributed_utils.using_backend(distributed_utils.HorovodBackend):
    #     data_sampler = torch.utils.data.distributed.DistributedSampler(
    #         ds, num_replicas=distr_backend.get_world_size(), rank=distr_backend.get_rank()
    #     )
    # else:
    data_sampler = None

    dl_train = DataLoader(
        ds,
        config.batch_size,
        shuffle=data_sampler is None,
        sampler=data_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(config.num_workers > 0),
    )

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to('cuda')
    model.resize_token_embeddings(len(ds.cards.tokenizer))

    # Testing Model input
    for input_ids, attention_mask, ground_truth in dl_train:
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        
        model.generate(input_ids=input_ids, 
                       attention_mask=attention_mask,
                       max_new_tokens=60,
                       temperature=0.9,
                       pad_token_id=ds.cards.tokenizer.pad_token_id)
        print()
        break

    resume = False
    if config.Resume.chk_ptn != "":
        print("--> Loading data from: {}".format(config.Resume.chk_ptn))
        state_dict = torch.load(config.Resume.chk_ptn, map_location="cpu")["weights"]
        model.load_state_dict(state_dict)
        resume = True

    if not False: #using_deepspeed:
        model = model.cuda()

    assert len(ds) > 0, "folder does not contain any decks"
    # if distr_backend.is_root_worker():
    print(f"{len(ds)} decks found for training")

    opt = AdamW(model.parameters(), **config.optimizer)
    step_size_up = int(config.coeff_step_size_up * len(dl_train))
    step_size_down = int(config.coeff_step_size_down * len(dl_train))
    # New schedular.
    sched = CyclicLR(
        optimizer=opt,
        step_size_up=step_size_up,
        step_size_down=step_size_down,
        **config.schedular
    )
    print()

main()