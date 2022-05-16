from flask import Flask, request, jsonify, render_template
import torch 
import torch.nn as nn
import torch.optim as optim 
from model import Encoder, Decoder, Seq2Seq
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from tqdm import tqdm 
from dataloader import get_loader
from dataset import TranslateDataset
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})

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
encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 512
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

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

checkpoint = torch.load("output/my_checkpoint.pth")

model.load_state_dict(checkpoint['state_dict'])



@app.route('/predict', methods=['POST'])
def results():
    data = request.get_json(force=True)
    text=data["message"]
    output = translate_sentence(model,text,trainDataset.eng_vocab,trainDataset.hin_vocab,device)
    return jsonify({"original_text":text,"transliterated_text":''.join(output)})

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    # app.run(debug=False)
    app.run(threaded=True, port=5000)