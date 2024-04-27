import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import List, Tuple


class DjangoDataset(Dataset):
    def __init__(self, code_data: List[str], pseudo_data: List[str]):
        self.code_data = code_data
        self.pseudo_data = pseudo_data

    def __len__(self):
        return len(self.code_data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.code_data[idx], self.pseudo_data[idx]


def load_datasets():
    # Load datasets from Hugging Face's datasets
    django_train = load_dataset("AhmedSSoliman/DJANGO", split="train")
    django_test = load_dataset("AhmedSSoliman/DJANGO", split="test")
    django_all = load_dataset("AhmedSSoliman/DJANGO", split="all")

    conala_train = load_dataset("neulab/conala", split="train")
    conala_test = load_dataset("neulab/conala", split="test")
    conala_all = load_dataset("neulab/conala", split="all")

    # Ensure datasets are formatted as PyTorch tensors
    for dataset in [django_train, django_test, django_all, conala_train, conala_test, conala_all]:
        dataset.with_format("torch")

    return {
        "django_train": django_train,
        "django_test": django_test,
        "django_all": django_all,
        "conala_train": conala_train,
        "conala_test": conala_test,
        "conala_all": conala_all
    }


def get_training_data():
    datasets = load_datasets()
    django_all = datasets["django_all"]
    conala_all = datasets["conala_all"]

    # Combine and filter data
    combined_data_code = django_all["code"] + conala_all["snippet"]
    combined_data_pseudo = django_all["nl"] + [pseudo for pseudo in conala_all["rewritten_intent"] if
                                               pseudo is not None]
    return combined_data_code, combined_data_pseudo


def create_data_iterators():
    code_data, pseudo_data = get_training_data()
    return iter(code_data), iter(pseudo_data)


def create_dataloaders(batch_size: int):
    datasets = load_datasets()
    train_dataset = DjangoDataset(datasets["django_train"]["code"], datasets["django_train"]["nl"])
    val_dataset = DjangoDataset(datasets["django_test"]["code"], datasets["django_test"]["nl"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_dataloader, val_dataloader


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    from tokenizer import text_transform, SRC_LANGUAGE, TGT_LANGUAGE, PAD_IDX
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
