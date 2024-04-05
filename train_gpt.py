import numpy as np
from torch.utils.data import DataLoader

# Dataset Config
from configs import config
from configs import parse_args
from dataset import mtg_decks

# Model
from transformers import AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from model import GPTNeoForMTG


def main():
    
    args = parse_args()
    

    ds = mtg_decks(**config.deckDataset, cards=config.cardDataset)

    model_config = AutoConfig.from_pretrained(config.model.name)
    model = GPTNeoForMTG(model_config, ds.masked_amount, ds.cards.all_cards)
    model.resize_token_embeddings(len(ds.cards.tokenizer))


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=ds.cards.tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        save_steps=1000,
        save_total_limit=25,
    )

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds,
    )

    # Train the model
    trainer.train()
    
    print()

main()