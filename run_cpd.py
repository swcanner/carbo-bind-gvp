import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default='./models/',
                    help='directory to save trained models, default=./models/')
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                   help='number of threads for loading data, default=4')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='max number of nodes per batch, default=3000')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='training epochs, default=100')
parser.add_argument('--cath-data', metavar='PATH', default='./data/chain_set.jsonl',
                    help='location of CATH dataset, default=./data/chain_set.jsonl')
parser.add_argument('--cath-splits', metavar='PATH', default='./data/chain_set_splits.json',
                    help='location of CATH split file, default=./data/chain_set_splits.json')
parser.add_argument('--ts50', metavar='PATH', default='./data/str_chain_json.json',
                    help='location of TS50 dataset, default=./data/str_chain.json')
parser.add_argument('--train', action="store_true", help="train a model")
parser.add_argument('--test-r', metavar='PATH', default=None,
                    help='evaluate a trained model on recovery (without training)')
parser.add_argument('--test-p', metavar='PATH', default=None,
                    help='evaluate a trained model on perplexity (without training)')
parser.add_argument('--n-samples', metavar='N', default=1,
                    help='number of sequences to sample (if testing recovery), default=100')
parser.add_argument('--predict', metavar='PATH', default=None,
                    help='predict trained model on CATH dataset')

args = parser.parse_args()
assert sum(map(bool, [args.train, args.test_p, args.test_r,args.predict])) == 1, \
    "Specify exactly one of --train, --test_r, --test_p, --predict"

import torch
import torch.nn as nn
import gvp.data, gvp.models
from datetime import datetime
import tqdm, os, json
import numpy as np
from sklearn.metrics import confusion_matrix
import torch_geometric
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
print = partial(print, flush=True)

node_dim = (100, 16)
edge_dim = (32, 1)
NUM_SCALARS = 6+21
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(args.models_dir): os.makedirs(args.models_dir)
model_id = int(datetime.timestamp(datetime.now()))
dataloader = lambda x: torch_geometric.data.DataLoader(x,
                        num_workers=args.num_workers,
                        batch_sampler=gvp.data.BatchSampler(
                            x.node_counts, max_nodes=args.max_nodes))
predict_dataloader = lambda x: torch_geometric.data.DataLoader(x, batch_size=1);

def main():

    #model = gvp.models.CPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    model = gvp.models.CPDModel((NUM_SCALARS, 3), node_dim, (32, 1), edge_dim).to(device)
    #for name, param in model.named_parameters():
    #    print(name, param.data.shape)

    print("Loading CATH dataset")
    #cath = gvp.data.CATHDataset(path="data/chain_set.jsonl",
    #                            splits_path="data/chain_set_splits.json")

    cath = gvp.data.CATHDataset(path="data/str_chain_json.json",
                                splits_path="data/str_chain_split.json")
    #cath = gvp.data.CATHDataset(path="data/str_json.json",
#                                splits_path="data/str_75-10-15_split.json")

    trainset, valset, testset = map(gvp.data.ProteinGraphDataset,
                                    (cath.train, cath.val, cath.test))

    if args.test_r or args.test_p:
        #ts50set = gvp.data.ProteinGraphDataset(json.load(open(args.ts50)))
        ts50set = gvp.data.CATHDataset(path="data/str_chain_json.json",
                                    splits_path="data/str_chain_split.json")
        model.load_state_dict(torch.load(args.test_r or args.test_p))

    if args.test_r:
        print("Testing on CATH testset"); test_recovery(model, testset)
        print("Testing on TS50 set"); test_recovery(model, ts50set)

    elif args.test_p:
        print("Testing on CATH testset"); test_perplexity(model, testset)
        print("Testing on TS50 set"); test_perplexity(model, ts50set)

    elif args.train:
        train(model, trainset, valset, testset)

    elif args.predict:
        print("Predicting on CATH dataset:");
        model.load_state_dict(torch.load(args.predict))
        predict(model, trainset, valset, testset)


def train(model, trainset, valset, testset):
    train_loader, val_loader, test_loader = map(dataloader,
                    (trainset, valset, testset))

    epoch_num = [];
    loss_epoch_train = [];
    loss_epoch_val = [];
    optimizer = torch.optim.Adam(model.parameters())
    best_path, best_val = None, np.inf
    lookup = train_loader.dataset.num_to_letter
    for epoch in range(args.epochs):
        model.train()
        loss, acc, confusion = loop(model, train_loader, optimizer=optimizer, loss_fn=dice_loss)
        path = f"{args.models_dir}/{model_id}_{epoch}.pt"
        torch.save(model.state_dict(), path)
        print(f'EPOCH {epoch} TRAIN loss: {loss:.4f} acc: {acc:.4f}')
        print(confusion)
        #print_confusion(confusion, lookup=lookup)
        epoch_num.append(epoch)
        loss_epoch_train.append(loss)

        model.eval()
        with torch.no_grad():
            loss, acc, confusion = loop(model, val_loader,loss_fn=dice_loss)
        print(f'EPOCH {epoch} VAL loss: {loss:.4f} acc: {acc:.4f}')
        #print_confusion(confusion, lookup=lookup)
        print(confusion)

        loss_epoch_val.append(loss)
        if loss < best_val:
            best_path, best_val = path, loss
        print(f'BEST {best_path} VAL loss: {best_val:.4f}')

        plt.plot(epoch_num,loss_epoch_train,'b-',label="Train")
        plt.plot(epoch_num,loss_epoch_val,'r-',label="Val")
        plt.legend()
        path = f"{args.models_dir}/{model_id}_all-chain.png"
        plt.savefig(path,transparent=True,dpi=300)
        plt.clf()

        #save it to a different file as well
        boi = np.array([epoch_num,loss_epoch_train,loss_epoch_val])
        path = f"{args.models_dir}/{model_id}_all-chain.npy"
        np.save(path,boi)

    print(f"TESTING: loading from {best_path}")
    model.load_state_dict(torch.load(best_path))

    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, test_loader)
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')
    print(confusion)
    #print_confusion(confusion,lookup=lookup)

def test_perplexity(model, dataset):
    model.eval()
    with torch.no_grad():
        loss, acc, confusion = loop(model, dataloader(dataset))
    print(f'TEST perplexity: {np.exp(loss):.4f}')
    print_confusion(confusion, lookup=dataset.num_to_letter)

def test_recovery(model, dataset):
    recovery = []

    for protein in tqdm.tqdm(dataset):
        protein = protein.to(device)
        h_V = (protein.node_s, protein.node_v)
        h_E = (protein.edge_s, protein.edge_v)
        sample = model.sample(h_V, protein.edge_index,
                              h_E, n_samples=args.n_samples)
        #print(sample.shape)

        recovery_ = sample.eq(protein.Y).float().mean().cpu().numpy()
        recovery.append(recovery_)
        print(protein.name, recovery_, np.sum(sample.float().cpu().numpy()), dice_loss(sample,protein.Y), flush=True)

    recovery = np.median(recovery)
    print(f'TEST recovery: {recovery:.4f}')


def predict(model, trainset, valset, testset):
    train_loader, val_loader, test_loader = map(predict_dataloader,
                    (trainset, valset, testset))


    """
    alfa, beta, gamma = pred_loader(model,train_loader)
    boi = np.array([alfa,beta])
    path = f"./data/predictions_train.npy"
    np.save(path,boi)
    boi = np.array(gamma)
    path = f"./data/predictions_train_name.npy"
    np.save(path,boi)

    alfa, beta, gamma = pred_loader(model,val_loader)
    boi = np.array([alfa,beta])
    path = f"./data/predictions_val.npy"
    np.save(path,boi)
    boi = np.array(gamma)
    path = f"./data/predictions_val_name.npy"
    np.save(path,boi)
    """
    alfa, beta, gamma = pred_loader(model,test_loader)
    boi = np.array([alfa,beta])
    path = f"./data/predictions_test.npy"
    np.save(path,boi)
    boi = np.array(gamma)
    path = f"./data/predictions_test_name.npy"
    np.save(path,boi)



    #print_confusion(confusion,lookup=lookup)

def pred_loader(model, loader):
    name = [];
    n = 0;
    out_pred = np.zeros((600,25000)) - 1;
    out_y = np.zeros((600,25000)) - 1;

    with torch.no_grad():
        for batch in loader:


            print(n,batch.name)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)

            #print("HV:",h_V[0].shape,h_V[1].shape)
            #print("HE:",h_E[0].shape,h_E[1].shape)

            logits = model(h_V, batch.edge_index, h_E, label=batch.Y)
            logits, label, title = logits, batch.Y, batch.name[0]

            #print(title,logits.shape,label.shape)
            logits = logits.detach().cpu().numpy()
            #print(title,logits[:,0].shape,label.shape)

            out_pred[n,:len(logits)] = logits[:,0]
            out_y[n,:len(logits)] = label.detach().cpu().numpy()
            name.append(title)
            n += 1;
            break;

            #print(name)

    return out_pred[:n,:], out_y[:n,:], name;


def loop(model, dataloader, optimizer=None, loss_fn=None):

    #confusion = np.zeros((20, 20))
    confusion = np.zeros((2,2))
    t = tqdm.tqdm(dataloader)
    if (loss_fn == None):
        loss_fn = nn.MSELoss()

    total_loss, total_correct, total_count = 0, 0, 0

    for batch in t:
        if optimizer: optimizer.zero_grad()

        batch = batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)

        #print("HV:",h_V[0].shape,h_V[1].shape)
        #print("HE:",h_E[0].shape,h_E[1].shape)

        logits = model(h_V, batch.edge_index, h_E, label=batch.Y)
        #logits, seq = logits[batch.mask], batch.seq[batch.mask]
        logits, label = logits[batch.mask], batch.Y[batch.mask]
        label = label.unsqueeze(1)
        #print("logits:",logits.shape)
        #print("label:",label.shape)
        #print(logits,label)

        loss_value = loss_fn(logits, label).float();
        #print(loss_value)
        if optimizer:
            loss_value.backward()
            optimizer.step()

        num_nodes = int(batch.mask.sum())
        total_loss += float(loss_value) * num_nodes
        total_count += num_nodes
        pred = (logits.detach().cpu().numpy() > .5) * 1;
        #print(np.sum(pred==0),np.sum(pred==1))
        true = label.detach().cpu().numpy()
        #print(np.sum(true==0),np.sum(true==1))
        total_correct += (pred == true).sum()
        #print(pred,"\n\n\n",true)
        #print(pred == true)
        #print(true.size,total_correct)
        confusion += confusion_matrix(true, pred, labels=range(2))
        #print(confusion)
        t.set_description("%.5f" % float(total_loss/total_count))

        torch.cuda.empty_cache()

    return total_loss / total_count, total_correct / total_count, confusion

def print_confusion(mat, lookup):
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = '\n'
    for i in range(2):
        res += '\t{}'.format(lookup[i])
    res += '\tCount\n'
    for i in range(2):
        res += '{}\t'.format(lookup[i])
        res += '\t'.join('{}'.format(n) for n in mat[i])
        res += '\t{}\n'.format(sum(counts[i]))
    print(res)


def dice(y_pred, y_true, smoothing_factor=0.01):
    #print("y_true:",y_true.shape,y_true)
    #y_true_f = torch.flatten(y_true,1)
    #print(y_true_f)
    #print("y_pred:",y_pred.shape,y_pred)
    #y_pred_f = torch.flatten(y_pred,1)
    # print(y_true_f, y_pred_f,"----")
    y_pred_f = y_pred
    y_true_f = y_true
    #print("y_true:",y_true.shape,y_true)
    #print("y_pred:",y_pred.shape,y_pred,y_pred_f)

    intersection = torch.sum(y_true_f * y_pred_f)
    #print(y_true_f * y_pred_f)
    return ((2. * intersection + smoothing_factor)
            / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smoothing_factor))

def dice_loss(y_pred, y_true):
    #print(y_true.shape, y_pred.shape)
    return 1-dice(y_pred, y_true)

if __name__== "__main__":
    main()
