import numpy as np
import torch
import os
from torch.utils.data import random_split
import wandb
from dotenv import load_dotenv, find_dotenv

# Dataset Config
from configs import config
from configs import parse_args
from dataset import mtg_decks
from sklearn.model_selection import train_test_split

# Model
from transformers import AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPTNeoModel, SchedulerType
from model import GPTNeoForMTG, CustomTrainer

def compute_metrics(eval_pred):
    (logits, choices, *_), labels = eval_pred
    logits = logits.squeeze(1)
    choices = choices.flatten()

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    top1_preds = torch.topk(probs, k=1).indices.squeeze(-1)
    top5_preds = torch.topk(probs, k=5).indices
    top15_preds = torch.topk(probs, k=15).indices

    # Check if true labels are within predictions
    top1_correct = top1_preds == torch.tensor(choices)
    top5_correct = torch.tensor([choices[i] in top5_preds[i] for i in range(len(choices))])
    top15_correct = torch.tensor([choices[i] in top15_preds[i] for i in range(len(choices))])


    top1_accuracy = top1_correct.float().mean().item()
    top5_accuracy = top5_correct.float().mean().item()
    top15_accuracy = top15_correct.float().mean().item()
    # print(top1_accuracy, top5_accuracy)
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'top15_accuracy': top15_accuracy,
    }


def main():
    wandb.init(project='huggingface', resume='must', id='s7fnr6xq')
    
    args = parse_args()
    _ = load_dotenv(find_dotenv()) 
    # wandb.login(key=os.getenv('WANDB_API_KEY'))

    ds = mtg_decks(**config.deckDataset, cards=config.cardDataset)
    train_set, test_set = random_split(ds, [0.90, 0.10], generator=torch.Generator().manual_seed(42))

    model_config = AutoConfig.from_pretrained(config.model.name)

    if config.Resume.chk_ptn:
        model = GPTNeoForMTG.from_pretrained(config.Resume.chk_ptn,
                                            tokenizer_length=len(ds.cards.tokenizer), 
                                            card_vocab=len(ds.cards.all_cards)
                                            )
    else:
        model = GPTNeoForMTG(config=model_config, 
                            tokenizer_length=len(ds.cards.tokenizer), 
                            card_vocab=len(ds.cards.all_cards)
                            )

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=ds.cards.tokenizer, mlm=True
    # )

    training_args = TrainingArguments(
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="epoch",
        # eval_steps=11,
        warmup_ratio = 0.05,
        lr_scheduler_type=SchedulerType.COSINE,
        overwrite_output_dir=True,
        output_dir="./output",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        dataloader_num_workers=config.num_workers,
        save_steps=2000,
        save_total_limit=30,
        resume_from_checkpoint=config.Resume.chk_ptn,
        report_to="wandb",
    )

    # Create the trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=None,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=config.Resume.chk_ptn)
    
    print()


if __name__ == '__main__':
    main()