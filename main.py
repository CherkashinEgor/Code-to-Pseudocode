import torch
from train import train_model
from model import Seq2SeqTransformer, load_model
from data import create_dataloaders
from tokenizer import train_tokenizers, load_tokenizers
from translate import translate, translate_with_sampling

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_code, tokenizer_pseudo = load_tokenizers()

    # Load data
    train_dataloader, val_dataloader = create_dataloaders(batch_size=128)

    # Initialize the model
    SRC_VOCAB_SIZE = len(tokenizer_code.get_vocab())
    TGT_VOCAB_SIZE = len(tokenizer_pseudo.get_vocab())

    # model = Seq2SeqTransformer(
    #     num_encoder_layers=3,
    #     num_decoder_layers=3,
    #     emb_size=512,
    #     nhead=8,
    #     src_vocab_size=SRC_VOCAB_SIZE,
    #     tgt_vocab_size=TGT_VOCAB_SIZE,
    #     dim_feedforward=2048,
    #     dropout=0.1
    # ).to(device)

    # Train the model
    # num_epochs = 10
    # train_model(model, train_dataloader, val_dataloader, num_epochs, device)

    model = load_model("checkpoints/model_checkpoint_epoch_18 (1).pth", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, device)

    # Translate a code snippet
    code = "def add(a, b):\n    return a + b"
    print(translate(model, code, device))
    print(translate_with_sampling(model, code, device, temperature=0.5))

if __name__ == '__main__':
    main()
