from dataset import mtg_cards

import os
import torch
import json
import numpy as np
import os.path as osp
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class mtg_decks(Dataset):
    def __init__(
        self,
        root="./data",
        json_name="AllDecks.npy",
        deck_length=60,
        mask_percent=0.3,
        cards=None,
        **kwargs,
    ):
        if isinstance(root, str):
            self.dataset_root = root

        self.deck_length = deck_length
        self.masked_amount = round(deck_length * mask_percent)
        self.cards = mtg_cards(**cards)
        self.db_file = json_name
        self.db_file = osp.join(self.dataset_root, self.db_file)

        self.deckbase = []
        if osp.exists(self.db_file):
            self.deckbase = np.load(self.db_file)

        else:
            self._get_db()

            self.deckbase = np.array(self.deckbase)
            np.save(self.db_file, self.deckbase)


    def _get_db(self):
        deck_folders = next(os.walk(self.dataset_root))[1]

        for Folder in deck_folders:
            path = osp.join(self.dataset_root, Folder)
            eval('self.'+Folder)(path)
        pass

    def mtg_official(self, path):
        # return
        decks = [osp.join(path, name) for name in os.listdir(path) if name.split('.')[-1] == 'json']
        
        for deck_name in tqdm(
            decks,
            desc='MTG Offical Decks',
        ):

            card_names = []
            with open(deck_name, 'r', encoding='utf-8') as file:
                deck = json.load(file)['data']
            
            mainboard = deck['mainBoard']

            for card in mainboard:
                if not self.cards.check_database(card['name']):
                    continue

                card_names += [card['name']] + ["[MASK]"] * (card['count']-1)

            if len(card_names) == self.deck_length:
                self.deckbase.append(card_names)
            

    def mtg_top8(self, path):
        decks = [osp.join(path, name) for name in os.listdir(path) if name.split('.')[-1] == 'txt']
        
        for deck_name in tqdm(
            decks,
            desc='MTG Top8 Decks',
        ):
            with open(deck_name, 'r') as file:
                deck = file.readlines()

            card_names = []
            for line in deck:
                line = line.strip()
                if line == 'Sideboard':
                    break

                count, *name_parts = line.split(" ")
                count = int(count)
                name = " ".join(name_parts).split("/")[0].strip()

                if not self.cards.check_database(name):
                    break
                
                card_names.extend([name] + ["[MASK]"] * (count-1))

            if len(card_names) == self.deck_length:
                self.deckbase.append(card_names)


    def __len__(self):
        return self.deckbase.shape[0]
    

    def __getitem__(self, idx):
        deck = self.deckbase[idx]
        deck = deck[np.random.permutation(len(deck))]

        mask = np.array([True] * self.masked_amount + [False] * (len(deck) - self.masked_amount))
        mask = mask[np.random.permutation(len(deck))]

        gt = deck[mask]
        # deck[mask] = self.cards.token_ids.mask_token

        tokenized_output = self.cards.return_card_batch(deck)
        
        # Remove padding from each card
        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask'].bool()
        input_ids = input_ids[attention_mask]
        
        return input_ids

    
    def max_length(self):
        lengths = []

        for j in range(1):
            for i in tqdm(range(self.__len__())):
                lengths.append(len(self.__getitem__(i)))

        
        data = np.array(lengths)
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=range(np.min(data), np.max(data) + 2), align='left', color='skyblue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Tokens per Deck')
        plt.grid(axis='y', alpha=0.75)

        # Show the plot
        plt.show()
        
        return max(lengths)