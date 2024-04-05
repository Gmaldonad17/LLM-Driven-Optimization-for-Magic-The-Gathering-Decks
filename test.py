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


# model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B').to('cuda')

# # Assuming 'tokenizer_name' should be the string name or path of the tokenizer
# tokenizer_name = 'EleutherAI/gpt-neo-2.7B'
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# # token_ids = {
# #     'additional_special_tokens': ['[CLONE]', '[MASK]', '[SEP]'],
# #     'pad_token': '[PAD]'
# # }
# # Adding special tokens to the tokenizer
# # tokenizer.add_special_tokens(token_ids)

# # Preparing the sample text
# sample = tokenizer("I stuck my balls in her", return_tensors='pt')
# sample['input_ids'] = sample['input_ids'].to('cuda')
# sample['attention_mask'] = sample['attention_mask'].to('cuda')

# # Generating tokens with specified parameters
# model.resize_token_embeddings(len(tokenizer))
# gen_tokens = model.generate(
#     **sample,
#     max_new_tokens=350,  # Example parameter: maximum length of the generated sequence
#     do_sample=True,
#     # num_beams=5,    # Example parameter: using beam search with 5 beams
#     temperature=0.9,  # Example parameter: sampling temperature
#     pad_token_id=tokenizer.pad_token_id  # Set pad token id for generation
# )

# # Decoding the generated tokens to text
# generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=False)
# print(generated_text)
# print()
# sample = decks.cards.tokenizer(sample, return_tensors='pt')
#     sample['input_ids'] = sample['input_ids'].to('cuda')
#     sample['attention_mask'] = sample['attention_mask'].to('cuda')
#     gen_tokens = model.generate(**sample, 
#                             do_sample=True,
#                             temperature=0.9,
#                             max_new_tokens=50,
#                             pad_token_id=decks.cards.tokenizer.pad_token_id,
#                             )
#     gen_text = decks.cards.tokenizer.batch_decode(gen_tokens[:, len(sample):])[0]



# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to('cuda')
# model = GPT2LMHeadModel.from_pretrained('gpt2-large').to('cuda')


def main():
    args = parse_args()
    decks = mtg_decks(**config.deckDataset, cards=config.cardDataset)

    # print(decks.cards.max_legnth())
    sample = decks.__getitem__(2)

    check = {}
    
    print(len(check))
    print(decks.max_length())
    print()
    

if __name__ == '__main__':
    main()