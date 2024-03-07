import json
import torch
from tqdm import tqdm
from configs import config
from configs import parse_args
from dataset import mtg_cards
from dataset import mtg_decks
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2TokenizerFast

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to('cuda')
# model = GPT2LMHeadModel.from_pretrained('gpt2-large').to('cuda')


def main():
    args = parse_args()
    decks = mtg_decks(**config.deckDataset, cards=config.cardDataset)

    # print(decks.cards.max_legnth())
    sample = decks.__getitem__(2).to('cuda')
    model.resize_token_embeddings(len(decks.cards.tokenizer))
    gen_tokens = model.generate(sample.unsqueeze(0), 
                            do_sample=True,
                            temperature=0.9,
                            max_length=1024,
                            pad_token_id=decks.cards.tokenizer.pad_token_id,
                            )
    gen_text = decks.cards.tokenizer.batch_decode(gen_tokens[:, len(sample):])[0]
    print(decks.max_length())
    print()
    

if __name__ == '__main__':
    main()