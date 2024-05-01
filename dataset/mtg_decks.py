from dataset import mtg_cards

import os
import random
import torch
import json
import numpy as np
import os.path as osp
from tqdm import tqdm
from copy import copy

import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class mtg_decks(Dataset):
    def __init__(
        self,
        root="./data",
        json_name="AllDecks.npy",
        deck_length=60,
        mask_amount=20,
        context_window=150,
        cards=None,
        **kwargs,
    ):
        if isinstance(root, str):
            self.dataset_root = root

        self.deck_length = deck_length
        self.masked_amount = mask_amount
        self.cards = mtg_cards(**cards)
        self.db_file = json_name
        self.db_file = osp.join(self.dataset_root, self.db_file)
        self.context_window = context_window
        self.max_deck_length = context_window - deck_length

        self.deckbase = []
        if osp.exists(self.db_file):
            self.deckbase = np.load(self.db_file)

        else:
            self._get_db()

            self.deckbase = np.array(self.deckbase)
            np.save(self.db_file, self.deckbase)

        self.prune_all_cards()
        # self.cards.max_deck_legnth = self.max_deck_length


    def _get_db(self):
        deck_folders = next(os.walk(self.dataset_root))[1]

        for Folder in deck_folders:
            path = osp.join(self.dataset_root, Folder)
            eval('self.'+Folder)(path)
        

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

                card_names += [card['name']] * (card['count']) # + ["[CLONE]"] 

            if len(card_names) == self.deck_length:
                self.deckbase.append(card_names)
            

    def mtg_top8(self, path, desc='MTG Top8 Decks'):
        decks = [osp.join(path, name) for name in os.listdir(path) if name.split('.')[-1] == 'txt']
        
        for deck_name in tqdm(
            decks,
            desc=desc,
        ):
            with open(deck_name, 'r') as file:
                deck = file.readlines()

            card_names = []
            for line in deck:
                line = line.strip()
                if line == 'Sideboard':
                    break
                if line == '':
                    continue

                count, *name_parts = line.split(" ")
                count = int(count)
                name = " ".join(name_parts).split("/")[0].strip()

                if not self.cards.check_database(name):
                    break
                
                card_names.extend([name] * (count))

            if len(card_names) == self.deck_length:
                self.deckbase.append(card_names)
        

    def mtg_decks(self, path):
        self.mtg_top8(path, desc='MTG.net Decks')


    def prune_all_cards(self):
        pruned_cards = {}
        index = 0  # Initialize an index counter outside the loop
        for idx in tqdm(range(self.__len__()), desc="Pruning Cards"):
            deck = self.deckbase[idx]
            for card in deck:
                if card in self.cards.all_cards.keys():
                    if card not in pruned_cards:  # Check if card is already added to avoid reindexing
                        pruned_cards[card] = copy(self.cards.all_cards[card])
                        pruned_cards[card]['index'] = index
                        index += 1  # Increment the index for each new card

        self.cards.all_cards = pruned_cards

    def __len__(self):
        return self.deckbase.shape[0]
    
    def _create_mask(self, deck):
        land_names = ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']
        # Find indices of land cards in the deck
        land_indices = [i for i, card in enumerate(deck) if card in land_names]

        # Initialize all positions as False initially
        mask = np.full(len(deck), False)

        # Determine indices to be masked, excluding land card indices
        if self.masked_amount <= (len(deck) - len(land_indices)):
            available_indices = [i for i in range(len(deck)) if i not in land_indices]
            masked_indices = np.random.choice(available_indices, self.masked_amount, replace=False)
            mask[masked_indices] = True
        else:
            raise ValueError("Masked amount exceeds the number of non-land cards in the deck")

        return mask

    def _mask_deck(self, deck):
        clone_token = '[CLONE]'

        mask = np.array([True] * self.masked_amount + \
                        [False] * (len(deck) - self.masked_amount))
        np.random.shuffle(mask)
        
        cards_idx = np.where(deck != clone_token)[0]

        for idx in cards_idx:
            if mask[idx]:  # Check if the card should be masked
                i = idx + 1
                while i < len(deck) and i not in cards_idx:
                    if not mask[i]:
                        mask[i] = True  # Update the mask to mask the clone token
                        mask[idx] = False  # Update the mask to unmask the original card
                        break
                    i += 1

        gt_idx = np.where(mask)[0]
        gt = []
        for idx in gt_idx:
            if idx not in cards_idx:
                card_idx = cards_idx[np.searchsorted(cards_idx, idx) - 1]
                gt.append(deck[card_idx])
            else:
                gt.append(deck[idx])

        deck = deck[~mask]
        gt = np.array(gt)
        # gt = torch.tensor([self.cards.all_cards[title]['index'] for title in gt])
        return deck, gt

    def _shuffle_deck(self, deck):
        np.random.shuffle(deck)
        return deck

        # clone_token = '[CLONE]'
        # cards_idx = np.where(deck != clone_token)[0]

        # parsed = np.split(deck, cards_idx)
        # np.random.shuffle(parsed)

        # return np.concatenate(parsed)

    def __getitem__(self, idx): # 25611 2115 length
        
        deck = self.deckbase[idx]

        deck = self._shuffle_deck(deck)
        mask = self._create_mask(deck)
        card_ids = torch.tensor([self.cards.all_cards[card]['index'] for card in deck])
        labels = deck[mask]
        labels = torch.tensor([self.cards.all_cards[title]['index'] for title in labels])

        tokenized_output = self.cards.return_card_batch(deck)
        tokenized_output['mask'] = mask
        tokenized_output['labels'] = labels
        tokenized_output['card_ids'] = card_ids

        return tokenized_output

    
    # def max_length(self):
    #     lengths = []

    #     for j in range(1):
    #         for i in tqdm(range(self.__len__())):
    #             lengths.append(sum(self.__getitem__(i)[1]))

        
    #     data = np.array(lengths)
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(data, bins=range(np.min(data), np.max(data) + 2), align='left', color='skyblue', edgecolor='black')
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     plt.title('Tokens per Deck')
    #     plt.grid(axis='y', alpha=0.75)

    #     # Show the plot
    #     plt.show()
        
    #     return max(lengths)