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
import statistics
from scipy.stats import uniform, loguniform

###############################################################################
# Parameter settings
###############################################################################

if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

#path = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
sel_fn = 'semihard_negative'
emsize = 512
nhead = 8
nhid = 512
#nlayers = 4
dropout = 0.1
#bs_train = 10
#bs_val = 8
#bs_test = 1
lr = 0.2
clip = 0.25
epochs = 25
log_interval = 10
#features = ["midipitch","duration","imaweight"]
margin = 1
warmup_epochs = 4

###############################################################################
# Dataloading
###############################################################################

corpus = inputProcessor.Corpus()
sam = sampler.TripletSelector()

def flush(data):
    return data.to(device)

    
def update_embeddings(transformer):
    #transformer.zero_grad()
    transformer.eval()
    start_time = time.time()
    for fam in corpus.samefamTrain.keys():
        for j in range(len(corpus.samefamTrain[fam])):
            with torch.no_grad():
                corpus.samefamTrain[fam][j]['Embedding'] = transformer(torch.tensor([corpus.samefamTrain[fam][j]['tokens']]).to(device)).squeeze(0)
    elapsed = time.time() - start_time
    print("Embedding calculations: {:5.2f} s".format(elapsed))

###############################################################################
# Train Loops
###############################################################################

def evaluate_online(transformer, batch_size, test=False):
# Turn on evaluation mode which disables dropout.
    criterion = nn.TripletMarginLoss(margin=margin)
    transformer.eval()
    total_loss = 0.
    seqLen = corpus.seqLen
    losses = []
    if test:
        data = corpus.samefamTest
        size = corpus.testsize
        batch_size = 1
    else:
        data = corpus.samefamValid
        size = corpus.validsize

    iterations = max(size // batch_size, 1)

    with torch.no_grad():
        for i in range(iterations):
            a,p,n,tfa,tfn = sam.sampleTriplets(data, batch_size)
            a,p,n = flush(a), flush(p), flush(n)
            a_out = transformer(a)
            p_out = transformer(p)
            n_out = transformer(n)
            loss = criterion(a_out, p_out, n_out).item()
            losses.append([loss, tfa[0], tfn[0]])
            total_loss += loss
    return total_loss / iterations, losses


def train_network_online(transformer, margin, lr, epoch, batch_size, optimizer, hard_triplets=False, sel_fn='semihard_negative'):
    # Turn on training mode which enables dropout.
    losses = []
    criterion = nn.TripletMarginLoss(margin=margin)
    transformer.train()
    #seqLen = corpus.seqLen # Shape wordt (seqLen, features)
    start_time = time.time()
    iterations = corpus.trainsize // batch_size
    log_interval = max(iterations // 10, 1)
    #iterations = 1
    total_triplets = 0
    for i in range(iterations):
        if hard_triplets:
            anchors,positives,negatives = [],[],[]
            datapoints = random.sample(corpus.trainMelodies, batch_size)
            triplets = sam.makeOnlineTriplets(datapoints, margin, sel_fn=sel_fn)
            total_triplets += len(triplets)
            for triple in triplets:
                anchors.append(datapoints[triple[0]]['tokens'])
                positives.append(datapoints[triple[1]]['tokens'])
                negatives.append(datapoints[triple[2]]['tokens'])
            a,p,n = torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)
            #a,p,n = corpus.trainMelodies[a]['tokens'], corpus.trainMelodies[p]['tokens'], corpus.trainMelodies[n]['tokens']
            #a,p,n,_,_ = sam.sampleFromIndices(num_samples=batch_size, replacement=False)
        else:
            a,p,n,_,_ = sam.sampleTriplets(corpus.samefamTrain, batch_size)
        a,p,n = flush(a), flush(p), flush(n)
        #transformer.zero_grad()
        optimizer.zero_grad()
        #print(po)
        a_out = transformer(a)
        p_out = transformer(p)
        n_out = transformer(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), clip)
        optimizer.step()

        # Copy updated parameters
        #for p in transformer.parameters():
            #if p != None and p.grad != None:
                #p.data.add_(p.grad, alpha=-lr)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #optimizer.step()
        #for name, param in transformer.named_parameters():
        #    if param.requires_grad:
        #        print(name)
        #print(optimizer.param_groups)

        cur_loss = loss.item()
        losses.append(cur_loss)
        
        elapsed = time.time() - start_time
        if i % log_interval == 0:
            print('| epoch {:3d} | batch {:3d} | {:5d} triplets | lr {:.2E} | ms/batch {:5.2f} | '
                        'loss {:5.4f}'.format(
                epoch, i+1, batch_size*(i+1), lr,
                elapsed * 1000 / log_interval, cur_loss))
        
        cur_loss = 0
        start_time = time.time()
    #trainlosses.append(statistics.mean(losses))
    return transformer, statistics.mean(losses)

def main(lr, batch_size, margin, nlayers, name, features, mode="incipit", load=-1, hard_triplets=False):

    print("Starting search with configuration: ")
    print("Learning rate: {:.2E}".format(lr))
    print("batch_size: {}".format(batch_size))
    print("margin: {}".format(margin))
    print("number of layers: {}".format(nlayers))
    
    if mode == 'incipit':
        path = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
    else:
        path = "../Thesis/Data/mtcfsinst2.0/mtcjson"

    corpus.readFolder(path, features)
    
    transformer = model.Transformer(src_vocab_size=10000, d_model=emsize, num_heads=nhead, num_layers=nlayers, d_ff=2048, max_seq_length=corpus.seqLen, dropout=dropout)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    warmupscheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1, total_iters=warmup_epochs)
    trainingLRscheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1, end_factor=0, total_iters=epochs - warmup_epochs)

    if load >= 0:
        #params = torch.load("Checkpoints/checkpoint_last_musicbert_base.pt", weights_only=False)  #MUSICBERT model
        #ptModel = torch.load("pretrain_model.ckpt", map_location='cpu')  #MUSICBERT model
        #print(ptModel)
        #roberta = transformer.load_state_dict(params['model'])
        #transformer.eval()
        #print(summary(params, params.shape))
        #optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
        while True: continue
        return

    try: 
        transformer.to(device)
        latest_val_loss = 1000
        best_val_loss = 1000
        val_losses, train_losses = [], []
        stagnate_counter = 0
        sel_fn = 'semihard_negative'
        for epoch in range(1, epochs + 1):

            epoch_start_time = time.time()
            #update_embeddings(transformer)
            transformer, train_losses = train_network_online(transformer, margin,lr, epoch, batch_size, optimizer, hard_triplets, sel_fn=sel_fn)
            val_loss = latest_val_loss
            latest_val_loss, losses = evaluate_online(transformer, batch_size, test=False)
            val_losses.append(latest_val_loss)
            if epoch < warmup_epochs:
                warmupscheduler.step()
                lr = warmupscheduler.get_last_lr()[0]
            else:
                if epoch == warmup_epochs:
                    trainingLRscheduler.step()
                    lr = trainingLRscheduler.get_last_lr()[0]
                    print("Warmup done")
                else:
                    trainingLRscheduler.step()
                    lr = trainingLRscheduler.get_last_lr()[0]

            if latest_val_loss >= val_loss and epoch >= 5:
                if not hard_triplets:
                    stagnate_counter += 1
                elif sel_fn == 'semihard_negative' and hard_triplets:
                    sel_fn = 'hardest_negative'
                else:
                    stagnate_counter += 1
            elif latest_val_loss < val_loss and epoch >= 5:
                stagnate_counter = 0


            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                latest_val_loss, math.exp(latest_val_loss)))
            print('-' * 89)
            if stagnate_counter >= 3:
                print("Stopping early at epoch {:3d}".format(epoch))
                break

            if val_loss < best_val_loss:
                with open("../Results/Experiments(feb2025)/{}.pt".format(name), 'wb') as f:
                    torch.save(transformer, f)
                    f.close()
                best_val_loss = val_loss

        with open("../Results/Experiments(feb2025)/{}Train.txt".format(name), "w") as f:
            f.write("CONFIGURATION: \n")
            f.write("Learning rate: {:.2E} \n".format(lr))
            f.write("margin: {} \n".format(margin))
            f.write("batch_size: {} \n".format(batch_size))
            f.write("number of layers: {} \n".format(nlayers))
            f.write("Train losses \n")
            for loss in train_losses:
                f.write(str(loss) + "\n")
            f.write("Val losses \n")
            for loss in val_losses:
                f.write(str(loss) + "\n")
            f.close()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("../Results/Experiments(feb2025)/{}.pt".format(name), 'rb') as f:
        transformer = torch.load(f, weights_only=False)
        transformer.eval()
        f.close()

    # Run on test data.
    test_loss, losses = evaluate_online(transformer, batch_size, test=True)

    with open("../Results/Experiments(feb2025)/{}Test.txt".format(name), 'w') as f:
        f.write("Test loss: " + str(test_loss) + "\n\n")
        f.write("Format: Loss TFAnchor TFNegative \n")
        for loss in losses:
            f.write(str(loss[0]) + " " + loss[1] + " " + loss[2] + "\n")
        f.close()

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == "__main__":
    main(5e-2, 768, 12, 12, 3072, 'adamNoClip', ["midipitch","duration","imaweight"], hard_triplets=False)