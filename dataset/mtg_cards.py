import os
import re
import os.path as osp
import json
import torch
from tqdm import tqdm
import numpy as np
from transformers import GPT2TokenizerFast, AutoTokenizer
from dataset import summarize_gpt

import matplotlib.pyplot as plt


class mtg_cards():
    def __init__(
        self,
        root="",
        raw="",
        db="",
        features=[],
        tokenizer='gpt2',
        max_tokens_card=300,
        token_ids={},
        summarize_model={},
        **kwargs,
    ):
        self.dataset_root = root
        self.features = features
        self.token_ids = token_ids
        self.desc_lengths = []
        self.max_tokens_card = max_tokens_card

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens(token_ids)

        self.db_file = osp.join(self.dataset_root, db)
        self.raw_data = osp.join(self.dataset_root, raw)

        if osp.exists(self.db_file):
            self.all_cards = json.load(open(self.db_file))

        else:
            self.all_cards = {}
            self.summarizer = summarize_gpt(**summarize_model)
            self._get_db()

            with open(self.db_file, 'w') as f:
                json.dump(self.all_cards, f)



    def _get_db(self,):
        with open(self.raw_data, 'r', encoding='utf-8') as file:
            data = json.load(file)['data']

        for set_code, set in tqdm(
            data.items(),
            desc='All Sets',
            ):

            for card in set['cards']:
                card['name'] = card['name'].split("//")[0].strip()

                if not self.check_card(card):
                    continue
                
                desc = self.create_card_summary(card)
                tokens = self.tokenizer(desc, return_tensors='pt')['input_ids'][0]
                if len(tokens) > self.max_tokens_card:
                    desc = self.summarizer.summarize_description(desc)
                
                self.all_cards[card['name']] = {'desc':desc, 'index':len(self.all_cards)}
        
        self.all_cards[self.token_ids.mask_token] = {'desc':self.token_ids.mask_token}


    def create_card_summary(self, card):
        # Initialize the base description with mandatory fields
        description = "{name} {type} {manaCost} {text}".format(
            name=card.get('name', '').strip(),
            type=card.get('type', '').strip(),
            manaCost=card.get('manaCost', '').strip(),
            text=card.get('text', '').strip(),
        )

        # Remove text within parentheses (including the parentheses)
        description = re.sub(r"\s*\([^)]*\)", "", description)

        # Conditionally append power and toughness if they are present
        if 'power' in card and 'toughness' in card:
            description += f" P/T {card['power'].strip()}/{card['toughness'].strip()}"

        # Clean up any double spaces and trailing spaces
        description = " ".join(description.split()).strip()

        return description
    

    def check_card(self, card):
        lang = card['language'] == 'English'
        record = card['name'] not in self.all_cards.keys()
        legal = 'modern' in card['legalities'].keys() and card['legalities']['modern'] == 'Legal'
        check = [lang, record]
        
        return all(check)
    
    
    def check_database(self, name):
        return self.all_cards.get(name, None)

    def return_card_text(self, name):
        return self.all_cards[name]['desc'] # + f" {self.token_ids.sep_token}"
    
    def return_card_token(self, name):
        return self.tokenizer(self.return_card_text(name), return_tensors='pt')
    
    def return_card_batch(self, deck):
        card_texts = [self.return_card_text(card) for card in deck]
        
        return self.tokenizer.batch_encode_plus(
            card_texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=None,
        )
    
    def max_legnth(self):
        lengths = []

        for i in tqdm(self.all_cards):
            desc = self.all_cards[i]['desc']
            tokens = self.tokenizer(desc, return_tensors='pt')['input_ids'][0]
            lengths.append(len(tokens))
            # if lengths[-1] > 100:
                # print(desc)
            

        data = np.array(lengths)
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=range(np.min(data), np.max(data) + 2), align='left', color='skyblue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Tokens Per Card')
        plt.grid(axis='y', alpha=0.75)

        # Show the plot
        plt.show()