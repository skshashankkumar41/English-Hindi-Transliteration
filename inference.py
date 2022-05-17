import torch 
import torch.nn as nn
import torch.optim as optim 
from model import Encoder, Decoder, Seq2Seq
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from tqdm import tqdm 
from dataloader import get_loader
from dataset import TranslateDataset
import argparse

torch.cuda.empty_cache()

num_epochs = 100
learning_rate = 0.001
batch_size = 32

trainLoader, valLoader, testLoader, trainDataset = get_loader(
    root_path = 'input/',
    batch_size = batch_size,
    prediction=True
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(trainDataset.eng_vocab)
input_size_decoder = len(trainDataset.hin_vocab)
output_size = len(trainDataset.hin_vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200

hidden_size = 256
num_layers = 1
enc_dropout = 0.4
dec_dropout = 0.4

step = 0 

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net, trainDataset.hin_vocab, device).to(device)

checkpoint = torch.load("output/my_checkpoint.pth",map_location=device)

model.load_state_dict(checkpoint['state_dict'])


def predict(model,sentence):
    output = translate_sentence(model,sentence,trainDataset.eng_vocab,trainDataset.hin_vocab,device)
    return ''.join(output)

# print("predict::",predict(model,"hello"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-msg", "--message", required=True, help="Message")

    args = vars(ap.parse_args())
    print(predict(model,args["message"]))