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
import query as q
#import clustervisualisation as cv


###############################################################################
# Parameter settings
###############################################################################

epochs = 25
log_interval = 10
warmup_epochs = 4

###############################################################################
# Dataloading
###############################################################################

corpus = inputProcessor.Corpus()
sam = sampler.TripletSelector()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def flush(data):
    return data.to(device)


def update_embeddings(transformer, data):
    transformer.eval()
    start_time = time.time()
    embs = []

    with torch.no_grad():
        embs = transformer(data.to(device))

    elapsed = time.time() - start_time
    print("Embedding calculations: {:5.2f} s".format(elapsed))
    return embs

###############################################################################
# Train Loops
###############################################################################

def evaluate_online(transformer, margin, batch_size, criterion, test=False):
# Turn on evaluation mode which disables dropout.
    transformer.eval()
    total_loss = 0.
    losses, tfs, anchorPositive, anchorNegative, positiveNegative = [], [], [], [], []
    if test:
        data = corpus.samefamTest
        size = corpus.testsize
    else:
        data = corpus.samefamValid
        size = corpus.validsize

    data = corpus.samefamTrain
    iterations = size

    with torch.no_grad():
        if not test:
            a,p,n,_,_ = sam.sampleTriplets(data, size)
            a,p,n = flush(a), flush(p), flush(n)
            a_out = transformer(a)
            p_out = transformer(p)
            n_out = transformer(n)
            loss = criterion(a_out, p_out, n_out).item()
            return loss
        else:
            for i in range(iterations):
                a,p,n,tfa,tfn = sam.sampleTriplets(data, 1)
                a,p,n = flush(a), flush(p), flush(n)
                a_out = transformer(a)
                p_out = transformer(p)
                n_out = transformer(n)
                loss = criterion(a_out, p_out, n_out).item()
                losses.append(loss)
                anchorPositive.append(torch.equal(a_out, p_out))
                anchorNegative.append(torch.equal(a_out, n_out))
                positiveNegative.append(torch.equal(n_out, p_out))
                tfs.append([tfa[0], tfn[0]])
            return  losses, tfs, anchorPositive, anchorNegative, positiveNegative


def train_network_online(transformer, margin, lr, epoch, batch_size, optimizer, criterion, labels, embs, hard_triplets=False, sel_fn='semihard_negative'):
    # Turn on training mode which enables dropout.
    losses = []
    triplet_calc_time = 0
    transformer.train()
    start_time = time.time()
    iterations = corpus.trainsize // batch_size
    log_interval = max(iterations // 10, 1)
    total_hard = 0
    #iterations = 1

    for i in range(iterations):
        if hard_triplets:
            batch_idx = random.sample(range(0, embs.size(0)), batch_size)
            start_time_triplets = time.time()
            triplets = sam.makeOnlineTriplets(embs[batch_idx], labels[batch_idx], margin, sel_fn=sel_fn)
            triplet_calc_time = time.time() - start_time_triplets
            total_hard += triplets.shape[0]
            #print(f"own hard triplets found: {len(triplets)} hard triplets found")
            a = corpus.trainMelodies[triplets[:,0]]
            p = corpus.trainMelodies[triplets[:,1]]
            n = corpus.trainMelodies[triplets[:,2]]
        else:
            a,p,n,_,_ = sam.sampleTriplets(corpus.samefamTrain, batch_size)
        a,p,n = flush(a), flush(p), flush(n)
        optimizer.zero_grad()
        a_out = transformer(a)
        p_out = transformer(p)
        n_out = transformer(n)
        loss = criterion(a_out, p_out, n_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=5)
        optimizer.step()

        cur_loss = loss.item()
        losses.append(cur_loss)
        
        elapsed = time.time() - start_time
        if i % log_interval == 0:
            if hard_triplets:
                print('| epoch {:3d} | batch {:3d} | {:5d} triplets | lr {:.2E} | ms/batch {:5.2f} | '
                        ' ms/hardtriplets {:5.2f} | loss {:5.4f}'.format(
                    epoch, i+1, total_hard, lr,
                    elapsed * 1000 / log_interval, triplet_calc_time * 1000 / log_interval, cur_loss))
            else:
                print('| epoch {:3d} | batch {:3d} | {:5d} triplets | lr {:.2E} | s/batch {:5.2f} | '
                        'loss {:5.4f}'.format(
                    epoch, i+1, batch_size*(i+1), lr,
                    elapsed * 1000 / log_interval, cur_loss))
        
        cur_loss = 0
        start_time = time.time()

    return transformer, losses

def main(params, name, features, mode="incipit", load=-1, hard_triplets=False):

    print(f"Device: {device}")

    print("Starting search with configuration: ")
    for param in params:
        print(param + ": " + str(params[param]))
    
    mAP = 0
    starting_lr = params['lr']
    lr = starting_lr
    criterion = nn.TripletMarginLoss(margin=params['margin'])
    dumpPath = f"../resultDump/{name}Dump.txt"
    open(dumpPath, 'w').close() # clear the file out

    if mode == 'incipit':
        path = "../Thesis/Data/mtcfsinst2.0_incipits(V2)/mtcjson"
    else:
        path = "../Thesis/Data/mtcfsinst2.0/mtcjson"

    # Check if data splits exists, if not make data split
    if 'trainData.json' not in os.listdir() or 'validData.json' not in os.listdir() or 'testData.json' not in os.listdir():
        print("Making data split")
        corpus.makeDataSplit(path)
    corpus.readData(features)
    
    transformer = model.Transformer(src_vocab_size=10000, d_model=params['d_model'], num_heads=params['n_heads'], num_layers=params['n_layers'], d_ff=params['d_ff'], max_seq_length=corpus.seqLen, dropout=params['dropout'])
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=params['epsilon'], weight_decay=params['wd'])
    warmupscheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1, total_iters=warmup_epochs)
    trainingLRscheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1, end_factor=0, total_iters=epochs - warmup_epochs)
    lr = warmupscheduler.get_last_lr()[0]

    if load >= 0:
        # implement loading code
        return

    try: 
        transformer.to(device)
        latest_val_loss, mAP_val = 999, 1
        best_val_loss = 1000
        val_losses, train_losses, train_embs = [], [], []
        stagnate_counter = 0
        starting_time = time.time()
        sel_fn = 'semihard_negative'
        #sel_fn = 'hardest_negative'
        train_labels = torch.tensor(corpus.trainLabels)
        valid_labels = torch.tensor(corpus.validLabels)
        test_labels = torch.tensor(corpus.testLabels)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            if hard_triplets:
                train_embs = update_embeddings(transformer, corpus.trainMelodies)

            transformer, train_losses_epoch = train_network_online(transformer, params['margin'], lr, epoch, params['batch_size'], optimizer, criterion, train_labels, train_embs, hard_triplets, sel_fn=sel_fn)
            train_losses.append(statistics.mean(train_losses_epoch))
            val_loss = latest_val_loss
            latest_val_loss = evaluate_online(transformer, params['margin'], params['batch_size'], criterion, test=False)
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

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                latest_val_loss, math.exp(latest_val_loss)))
            print('-' * 89)

            if latest_val_loss < best_val_loss:
                best_val_loss = val_loss
            
            with open(dumpPath, 'a') as f:
                f.write(f"Training_loss_epoch{epoch}:\n")
                for loss in train_losses_epoch:
                    f.write(f"{loss}\n")
                f.write(f"Validation_loss_epoch{epoch}:\n")
                f.write(f"{latest_val_loss}\n")
                f.close()
            
            if epoch % 1 == 0:
                mAP_train = q.main(update_embeddings(transformer, corpus.trainMelodies), train_labels, name, transformer, mode='Training', epoch=epoch)
                latest_mAP_val = q.main(update_embeddings(transformer, corpus.validMelodies), valid_labels, name, transformer, mode='Validation', epoch=epoch)
                
                with open(dumpPath, 'a') as f:
                    f.write(f"mAP value at epoch {epoch}: Train {mAP_train} Val {latest_mAP_val} \n")
                f.close()
                if latest_mAP_val < mAP_val and not hard_triplets:
                    print("Stopping training: MAP score not improved on validation set for last 10 epochs")
                    break
                elif latest_mAP_val < mAP_val and hard_triplets and sel_fn == 'semihard_negative':
                    sel_fn = 'hardest_negative'
                    print("Switching to hardest negative triplets")
                else:
                    mAP_val = latest_mAP_val
                    with open("../Weights/HyperparameterTuning/{}.pt".format(name), 'wb') as f:
                        torch.save(transformer, f)
                    f.close()

        with open("Results/Experiments(feb2025)/{}Train.txt".format(name), "w") as f:
            f.write("CONFIGURATION: \n")
            f.write(f"Training time: {round(time.time() - starting_time)} \n")
            for param in params:
                f.write(param + ": " + str(params[param]) + '\n')
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
    with open("../Weights/HyperparameterTuning/{}.pt".format(name), 'rb') as f:
        transformer = torch.load(f, weights_only=False)
        transformer.eval()
        f.close()

    # Run on test data.
    test_losses, tfs, anPo, anNe, poNe = evaluate_online(transformer, params['margin'], params['batch_size'], criterion, test=True)
    test_loss = statistics.mean(test_losses)
    with open(dumpPath, "a") as f:
        f.write(f"Test losses: TFA/TFN emAn emPo emNe:")
        for i in range(len(test_losses)):
            f.write(f"{test_losses[i]} {tfs[i][0]}/{tfs[i][1]} {anPo[i]} {anNe[i]} {poNe[i]} \n")
        f.close()

    with open("Results/Experiments(feb2025)/{}Test.txt".format(name), 'w') as f:
        f.write("Test loss: " + str(test_loss) + "\n\n")
        f.write("Format: Loss TFAnchor TFNegative \n")
        for i in range(len(tfs)):
            f.write(str(test_losses[i]) + " " + tfs[i][0] + " " + tfs[i][1] + "\n")
        f.close()

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)

    return q.main(update_embeddings(transformer, corpus.testMelodies), test_labels, name, transformer, mode="Test")

#if __name__ == "__main__":
    #main(5e-2, 768, 12, 12, 3072, 'TEST', ["midipitch","duration","imaweight"], hard_triplets=False)