# coding: utf-8
import argparse
import time
import math
import tempfile
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import torch
import inputProcessor
import model
import sampler
import random

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

###############################################################################
# Parameter settings
###############################################################################

if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

path = "../Thesis/Data/mtcfsinst2.0/mtcjson"
trainlosses = []
vallosses = []
emsize = 16
nhead = 4
nhid = 200
nlayers = 6
dropout = 0.2
bs_train = 50
bs_val = 10
bs_test = 1
lr = 0.2
clip = 0.50
epochs = 200
log_interval = 20
features = ["midipitch","duration","imaweight"]
margin = 1
dm = np.zeros((15000,15000),dtype=np.float32)
embs = {}

corpus = inputProcessor.Corpus()
sam = sampler.TripletSelector()

###############################################################################
# Dataloading
###############################################################################

def flush(data, bsize, seqlen):
    #data = data.narrow(0,0, bsize*seqlen)
    #data = torch.reshape(data, (19,100,2))
    #print(data[0])
    return data.to(device)

def get_batch(a, p, n, i, seqLen):
    '''
    anchors, positives, negatives = source[indices[0]][0]['x'], source[indices[0]][1]['x'], source[indices[0]][2]['x']
    for i in range(1, len(indices)):
        anchors = torch.cat((anchors, source[i][0]['x']), 0)
        positives = torch.cat((positives, source[i][1]['x']), 0)
        negatives = torch.cat((negatives, source[i][2]['x']), 0)
    '''
    batch_len = seqLen * bs_train
    a = a[i:i+batch_len]
    p = p[i:i+batch_len]
    n = n[i:i+batch_len]
    return a, p, n

###############################################################################
# Train Loop
###############################################################################
    
def update_embeddings(transformer):
    #transformer.zero_grad()
    transformer.eval()
    start_time = time.time()
    for i in range(len(corpus.samefam)):
        with torch.no_grad():
            corpus.samefam[i]['Embedding'] = transformer(torch.tensor([corpus.samefam[i][j]['tokens']]).to(device)).squeeze(0)
    elapsed = time.time() - start_time
    print("Embedding calculations: {:5.2f} s".format(elapsed))

def evaluate_online(transformer, batch_size, test=False):
# Turn on evaluation mode which disables dropout.
    criterion = nn.TripletMarginLoss(margin=margin)
    transformer.eval()
    total_loss = 0.
    seqLen = corpus.seqLen
    losses = []
    if test:
        data = corpus.samefamTest
        iterations = corpus.testsize // batch_size
    else:
        data = corpus.samefamValid
        iterations = corpus.validsize // batch_size

    with torch.no_grad():
        for i in range(iterations):
            a,p,n,tfa,tfn = sam.sampleTriplets(data, batch_size)
            a,p,n = flush(a, batch_size, seqLen), flush(p, batch_size, seqLen), flush(n, batch_size, seqLen)
            a_out = transformer(a)
            #a_out = a_out.view(-1, ntokens)
            p_out = transformer(p)
            #p_out = p_out.view(-1, ntokens)
            n_out = transformer(n)
            #n_out = n_out.view(-1, ntokens)
            loss = criterion(a_out, p_out, n_out).item()
            losses.append([loss, tfa[0], tfn[0]])
            total_loss += loss
    return total_loss / iterations, losses


def train_network_online(transformer, margin, lr, epoch, batch_size, hard_triplets=False):
    # Turn on training mode which enables dropout.
    criterion = nn.TripletMarginLoss(margin=margin)
    transformer.train()
    seqLen = corpus.seqLen # Shape wordt (seqLen, features)
    start_time = time.time()
    iterations = corpus.trainsize // batch_size
    if hard_triplets:
        triplets = sam.makeOnlineTriplets(corpus)
    for i in range(iterations):
        if hard_triplets:
            a,p,n,_,_ = random.sample(triplets, batch_size)
        else:
            a,p,n,_,_ = sam.sampleTriplets(corpus.samefamTrain, batch_size)
        a,p,n = flush(a, batch_size, seqLen), flush(p, batch_size, seqLen), flush(n, batch_size, seqLen)
        transformer.zero_grad()
        a_out = transformer(a)
        p_out = transformer(p)
        n_out = transformer(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), clip)

        # Copy updated parameters
        for p in transformer.parameters():
            if p != None and p.grad != None:
                p.data.add_(p.grad, alpha=-lr)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #optimizer.step()
        #for name, param in transformer.named_parameters():
        #    if param.requires_grad:
        #        print(name)
        #print(optimizer.param_groups)

        cur_loss = loss.item()
        trainlosses.append(cur_loss)
        
        elapsed = time.time() - start_time
        if i % log_interval == 0:
            print('| epoch {:3d} | batch {:3d} | {:5d}/{:5d} triplets | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.4f}'.format(
                epoch, i, batch_size*(i+1), iterations*batch_size, lr,
                elapsed * 1000 / log_interval, cur_loss))
        
        cur_loss = 0
        start_time = time.time()

def main(lr, d_model, nheads, n_layers, d_ff, name, features, mode="incipit", load=-1, hard_triplets=False):

    print("Starting search with configuration: ")
    print("Learning rate: {:5.4f}".format(lr))
    print("d_model: {}".format(d_model))
    print("n_heads: {}".format(nheads))
    print("n_layers: {}".format(n_layers))
    print("d_ff: {}".format(d_ff))
    best_val_loss = None
    corpus.readFolder(path, features)

    #encoder_layer = nn.TransformerEncoderLayer(d_model=emsize, nhead=nhead, dropout=dropout, device=device)
    transformer = model.Transformer(src_vocab_size=50000, d_model=d_model, num_heads=nheads, num_layers=n_layers, d_ff=d_ff, max_seq_length=corpus.seqLen, dropout=dropout)
    lr_delta = lr / 10
    
    if load >= 0:
        transformer = torch.load("Tuning Results2/{}funcCheck.pt".format(load))
        model.eval()
        print(model)
        return

    try: 
        transformer.to(device)
        latest_val_loss = 1000
        val_losses = []
        stagnate_counter = 0
        for epoch in range(1, epochs + 1):
            
            if epoch % (epochs // 10) == 0:
                lr -= lr_delta

            epoch_start_time = time.time()
            update_embeddings(transformer)
            train_network_online(transformer, margin, lr, epoch, bs_train, hard_triplets)
            val_loss = latest_val_loss
            latest_val_loss, losses = evaluate_online(transformer, bs_val, hard_triplets)
            val_losses.append(latest_val_loss)
            if len(val_losses) > 20 and latest_val_loss >= val_loss:
                stagnate_counter += 1
            else:
                stagnate_counter = 0

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                latest_val_loss, math.exp(latest_val_loss)))
            print('-' * 89)
            if stagnate_counter >= 5:
                print("Stopping early at epoch {:3d}".format(epoch))
                break

        with open("../Results/Experiments/{}Train.txt".format(name), "w") as f:
            f.write("CONFIGURATION: \n")
            f.write("Learning rate: {:5.2f} \n".format(lr))
            f.write("d_model: {} \n".format(d_model))
            f.write("nheads: {} \n".format(nheads))
            f.write("n_layers: {} \n".format(n_layers))
            f.write("d_ff: {} \n".format(d_ff))
            f.write("Train losses \n")
            for loss in trainlosses:
                f.write(str(loss) + "\n")
            f.write("Val losses \n")
            for loss in vallosses:
                f.write(str(loss) + "\n")
            f.close()

        if not best_val_loss or val_loss < best_val_loss:
            with open("../Results/Experiments/{}.pt".format(name), 'wb') as f:
                torch.save(transformer, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("../Results/Experiments/{}.pt".format(name), 'rb') as f:
        transformer = torch.load(f)
        f.close()

    # Run on test data.
    test_loss, losses = evaluate_online(transformer, bs_test, test=True)

    with open("../Results/Experiments/{}Test.txt".format(name), 'w') as f:
        f.write("Avg loss: " + str(test_loss) + "\n\n")
        f.write("Format: Loss TFAnchor TFNegative \n")
        for loss in losses:
            f.write(str(loss[0]) + " " + loss[1] + " " + loss[2] + "\n")
        f.close()

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == "__main__":
    main(0.3, 32, 4, 4, 256, 'running_example', ["midipitch","duration","imaweight"])