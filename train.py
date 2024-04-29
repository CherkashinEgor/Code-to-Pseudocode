import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import create_dataloaders
from model import Seq2SeqTransformer
from util import create_mask


def train_epoch(model, optimizer, train_dataloader, device):
    model.train()
    losses = 0
    for src, tgt in tqdm(train_dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = nn.CrossEntropyLoss(ignore_index=1)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, device):
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = nn.CrossEntropyLoss(ignore_index=1)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    return losses / len(val_dataloader)


def train_model(model, train_dataloader, val_dataloader, num_epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, optimizer, train_dataloader, device)
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")