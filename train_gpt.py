import numpy as np
import torch
import os
from torch.utils.data import DataLoader
# import wandb
from dotenv import load_dotenv, find_dotenv

# Dataset Config
from configs import config
from configs import parse_args
from dataset import mtg_decks

# Model
from transformers import AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPTNeoModel
from model import GPTNeoForMTG

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Assuming your labels are in the same format required by your logits predictions
    # Convert logits to probabilities and determine top-k predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    top1_preds = torch.topk(probs, k=1).indices.squeeze(-1)
    top5_preds = torch.topk(probs, k=5).indices

    # Check if true labels are within predictions
    top1_correct = top1_preds == torch.tensor(labels)
    top5_correct = torch.tensor([labels[i] in top5_preds[i] for i in range(len(labels))])

    top1_accuracy = top1_correct.float().mean().item()
    top5_accuracy = top5_correct.float().mean().item()

    return {
        'top1_accuracy': top1_accuracy * 100,
        'top5_accuracy': top5_accuracy * 100
    }


def main():
    
    args = parse_args()
    _ = load_dotenv(find_dotenv()) 
    # wandb.login(key=os.getenv('WANDB_API_KEY'))

    ds = mtg_decks(**config.deckDataset, cards=config.cardDataset)

    tokenizer_length = len(ds.cards.tokenizer)

    model_config = AutoConfig.from_pretrained(config.model.name)
    if config.Resume.chk_ptn:
        model = GPTNeoForMTG.from_pretrained(config.Resume.chk_ptn,
                                            tokenizer_length=tokenizer_length, 
                                            card_vocab=len(ds.cards.all_cards)
                                            )
    else:
        model = GPTNeoForMTG(config=model_config, 
                            tokenizer_length=tokenizer_length, 
                            card_vocab=len(ds.cards.all_cards)
                            )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=ds.cards.tokenizer, mlm=False
    )
    warmup_steps = int(config.num_train_epochs * ds.__len__() * 0.05 / config.batch_size)

    training_args = TrainingArguments(
        
        num_train_epochs=config.num_train_epochs,
        warmup_steps=warmup_steps,
        overwrite_output_dir=True,
        output_dir="./output",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=config.batch_size,
        dataloader_num_workers=config.num_workers,
        save_steps=2000,
        save_total_limit=30,
        resume_from_checkpoint=config.Resume.chk_ptn,
        report_to="wandb",
        # run_name="run-name",  # Optional: Specify a run name for a clear identification in wandb
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=config.Resume.chk_ptn)
    
    print()


if __name__ == '__main__':
    main()