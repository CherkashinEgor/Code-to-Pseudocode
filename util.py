import torch
import math
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import os

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones((sz, sz), device=device)).t()
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == 1).transpose(0, 1)
    tgt_padding_mask = (tgt == 1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def calculate_bleu_scores(generated_texts, reference_texts):
    tokenized_generated = [word_tokenize(generated) for generated in generated_texts]
    tokenized_references = [[word_tokenize(reference)] for reference in reference_texts]

    corpus_score = corpus_bleu(tokenized_references, tokenized_generated)
    sentence_scores = [sentence_bleu(refs, gen) for gen, refs in zip(tokenized_generated, tokenized_references)]
    average_sentence_score = sum(sentence_scores) / len(sentence_scores)
    return corpus_score, average_sentence_score

def calculate_rouge_scores(generated_texts, reference_texts):
    rouge = Rouge()
    scores = rouge.get_scores(generated_texts, reference_texts, avg=True)
    return scores

def calculate_semantic_similarity(texts1, texts2, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)

    similarity_scores = [cosine_similarities[i][i].item() for i in range(len(texts1))]
    return sum(similarity_scores) / len(similarity_scores) if len(similarity_scores) > 0 else 0

def save_model(model, path):
    #Check if folder exists, create if not
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), path)
