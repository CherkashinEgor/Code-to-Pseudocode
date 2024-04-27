from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import torch
from collections import OrderedDict
from torchtext.vocab import vocab
from typing import List

from data import create_data_iterators
import os


MAX_LENGTH = 70

# Tokenizer for code
tokenizer_code = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer_code.enable_padding(pad_id=1, pad_token="[PAD]", length=MAX_LENGTH)
tokenizer_code.enable_truncation(max_length=MAX_LENGTH)
tokenizer_code.pre_tokenizer = Whitespace()

# Tokenizer for pseudocode
tokenizer_pseudo = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer_pseudo.enable_padding(pad_id=1, pad_token="[PAD]", length=MAX_LENGTH)
tokenizer_pseudo.enable_truncation(max_length=MAX_LENGTH)
tokenizer_pseudo.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"])

# Assuming you have training data ready to train tokenizer, otherwise you need to set this up
code_iterator, pseudo_iterator = create_data_iterators()
tokenizer_code.train_from_iterator(code_iterator, trainer=trainer)
tokenizer_pseudo.train_from_iterator(pseudo_iterator, trainer=trainer)

# Vocabulary setup for the tokenizers-own
vocab_dict_code = tokenizer_code.get_vocab()
sorted_vocab_code = sorted(vocab_dict_code.items(), key=lambda x: x[1])
ordered_vocab_code = OrderedDict(sorted_vocab_code)

vocab_dict_pseudo = tokenizer_pseudo.get_vocab()
sorted_vocab_pseudo = sorted(vocab_dict_pseudo.items(), key=lambda x: x[1])
ordered_vocab_pseudo = OrderedDict(sorted_vocab_pseudo)

# Create torchtext's Vocab object
vocab_transform = {
    'code': vocab(ordered_vocab_code, specials=['[UNK]', '[PAD]', '[BOS]', '[EOS]']),
    'nl': vocab(ordered_vocab_pseudo, specials=['[UNK]', '[PAD]', '[BOS]', '[EOS]'])
}

# Token and vocab transformations
SRC_LANGUAGE = 'code'
TGT_LANGUAGE = 'nl'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class Tokenizer_own:
    def __init__(self, code_tokenizer, pseudo_tokenizer):
        self.code_tokenizer = code_tokenizer
        self.pseudo_tokenizer = pseudo_tokenizer

    def tokenize_code(self, s):
        return [token for token in self.code_tokenizer.encode(s).tokens]

    def tokenize_pseudo(self, s):
        return [token for token in self.pseudo_tokenizer.encode(s).tokens]

# Initialize with the correct objects
my_tokenizer = Tokenizer_own(tokenizer_code, tokenizer_pseudo)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# Set up the token transforms for both source and target languages
token_transform = {
    SRC_LANGUAGE: my_tokenizer.tokenize_code,
    TGT_LANGUAGE: my_tokenizer.tokenize_pseudo
}

# Adding the `text_transform` setup
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                               vocab_transform[ln],  # Numericalization
                                               tensor_transform)     # Add BOS/EOS and create tensor

def train_tokenizers():
    code_iterator, pseudo_iterator = create_data_iterators()
    tokenizer_code.train_from_iterator(code_iterator, trainer=trainer)
    tokenizer_pseudo.train_from_iterator(pseudo_iterator, trainer=trainer)
    return tokenizer_code, tokenizer_pseudo

def save_tokenizers(tokenizer_code, tokenizer_pseudo, directory='tokenizers-own-own'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    tokenizer_code.save(os.path.join(directory, 'tokenizer_code.json'))
    tokenizer_pseudo.save(os.path.join(directory, 'tokenizer_pseudo.json'))

def load_tokenizers(directory='tokenizers-own'):
    tokenizer_code = Tokenizer.from_file(os.path.join(directory, 'tokenizer_code.json'))
    tokenizer_pseudo = Tokenizer.from_file(os.path.join(directory, 'tokenizer_pseudo.json'))
    return tokenizer_code, tokenizer_pseudo

def train_or_load_tokenizers(directory='tokenizers-own'):
    if os.path.exists(directory):
        return load_tokenizers(directory)
    else:
        code_tok, pseudo_tok = train_tokenizers()
        save_tokenizers(code_tok, pseudo_tok)
        return code_tok, pseudo_tok

if __name__ == "__main__":
    # Train and save tokenizers-own
    code_tok, pseudo_tok = train_tokenizers()
    save_tokenizers(code_tok, pseudo_tok)
