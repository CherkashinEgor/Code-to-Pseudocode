import torch
from model import Seq2SeqTransformer
from util import generate_square_subsequent_mask
from tokenizer import text_transform, vocab_transform, SRC_LANGUAGE, TGT_LANGUAGE, EOS_IDX, BOS_IDX

def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str, device, max_len=50):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len, BOS_IDX, device).flatten()
    return " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(token) for token in tgt_tokens if token != EOS_IDX])

def temperature_sampling_decode(model, src, src_mask, max_len, start_symbol, temperature, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        prob = torch.nn.functional.softmax(prob / temperature, dim=-1)
        next_word = torch.multinomial(prob, 1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break

    return ys

def translate_with_sampling(model: torch.nn.Module, src_sentence: str, device, temperature=1.0, max_len=50):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
    tgt_tokens = temperature_sampling_decode(model, src, src_mask, max_len, BOS_IDX, temperature, device).flatten()
    return " ".join([vocab_transform[TGT_LANGUAGE].lookup_token(token) for token in tgt_tokens if token != EOS_IDX])

def generate_test_translations(model, code_test, device, model_name, all_generations):
    model.eval()
    translations = []
    translations_with_sampling = []
    with torch.no_grad():
        for src_sample in code_test:
            translation = translate(model, src_sample, device)
            translation_with_sampling = translate_with_sampling(model, src_sample, device, temperature=0.5)
            translations.append(translation)
            translations_with_sampling.append(translation_with_sampling)
    all_generations[model_name] = {"greedy": translations, "sampling": translations_with_sampling}
