import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import timeit
from tqdm import tqdm
import sklearn.metrics
import math

from boundary_dataset import AnchorDataset

torch.manual_seed(2)

def run_one_epoch(train_flag, dataloader, model, optimizer, device, dataset_size, epoch):

    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []    
    all_ys = []
    all_probs = []
    
    with tqdm(enumerate(dataloader), total=(dataset_size), unit="batch") as tepoch:
        for idx, (x_cnn, x_rnn, y) in tepoch: #tqdm.tqdm(enumerate(dataloader), total=dataset_size): # collection of tuples with iterator
            tepoch.set_description(f"Epoch {epoch+1}")
            (x_cnn, x_rnn, y) = ( x_cnn.type(torch.FloatTensor).to(device), x_rnn.type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device) ) # transfer data to GPU

            output = model(x_cnn, x_rnn) # forward pass
            output = output.squeeze() # remove spurious channel dimension
            loss = F.binary_cross_entropy( output, y ) # numerically stable

            if train_flag: 
                loss.backward() # back propagation
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.detach().cpu().numpy())
            accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )

            y_np = y.detach().cpu().numpy()
            probs_np = output.detach().cpu().numpy()
                    
            all_ys.extend(y_np.tolist())
            all_probs.extend(probs_np.tolist())
            
            tepoch.set_postfix(loss=loss.item(), batch_accuracy=100. * accuracy.detach().cpu().numpy())#, batch_auprc=100. * auprc)
            
    all_ys = np.array(all_ys)
    all_probs = np.array(all_probs)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(all_ys, all_probs)
    epoch_acc = np.mean( ( (all_probs > .5) == (all_ys > .5) ) )
    
    return( np.mean(losses), epoch_acc, precision, recall)


def train_model(model, train_data, validation_data, dataset_lengths, epochs=15, patience=4, verbose = True):
    train_length, val_length, test_length = dataset_lengths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    batch_size = 64
    train_dataset = AnchorDataset(train_data[0], train_data[1], 800, train_length)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers = 8, shuffle=True)
    
    val_dataset = AnchorDataset(validation_data[0], validation_data[1], 800, val_length)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers = 8)

    optimizer = torch.optim.RMSprop(model.parameters())
    
    train_accs = []
    val_accs = []
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'anchor_model_checkpoint_cnn_lstm.pt' # to save the best model fit to date
    for epoch in range(epochs):
        start_time = timeit.default_timer()
        
        train_loss, train_acc, train_pr, train_rec = run_one_epoch(True, train_dataloader, model, optimizer, device, math.ceil(len(train_dataset)/batch_size), epoch)
        val_loss, val_acc, val_pr, val_rec = run_one_epoch(False, val_dataloader, model, optimizer, device, math.ceil(len(val_dataset)/batch_size), epoch)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_loss < best_val_loss: 
            torch.save(model.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        
        if verbose == True:
            train_auprc = sklearn.metrics.auc(train_rec, train_pr)
            val_auprc = sklearn.metrics.auc(val_rec, val_pr)
            print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. auprc: %.4f. Val loss: %.4f acc: %.4f. auprc: %.4f. Patience left: %i" % 
                  (epoch+1, elapsed, train_loss, train_acc, train_auprc, val_loss, val_acc, val_auprc, patience_counter ))
    
    return model, train_accs, val_accs
