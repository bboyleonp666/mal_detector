import numpy as np
from time import time

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 128)
        self.lin2 = Linear(128, 128)
        self.lin = Linear(128, num_classes)

    def forward(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = gmp(x, batch)  # [batch_size, hidden_channels]
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        ## ignore softmax activation here, since we can obtain
        ## higher accuracy in our case
        x = F.softmax(x, dim=1)
        
        return x


class TorchTrainer:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
                
        self.arguments = locals()
        self.arguments['device'] = self.device
    
    def train(self, train_loader, valid_loader, epochs=20, save_path='model_saved/gcn_model.pt', verbose=True):
        self.arguments['epochs'] = epochs
        self.arguments['save_path'] = save_path
        
        train_acc  = np.zeros(epochs)
        train_loss = np.zeros(epochs)
        val_acc    = np.zeros(epochs)
        val_loss   = np.zeros(epochs)
        train_time = np.zeros(epochs)
        
        best_val_acc = 0
        for epoch in range(epochs):
            if verbose:
                epoch_start = f'Epoch ({epoch + 1}/{epochs})'
                print(epoch_start, end=' ')

            train_time[epoch] = self.train_epoch(train_loader)

            # evaluate the training accuracy and validation accuracy after each epoch
            train_acc[epoch], train_loss[epoch] = self.test(train_loader)
            val_acc[epoch], val_loss[epoch] = self.test(valid_loader)

            if val_acc[epoch] > best_val_acc:
                # save the best model according to validation accuracy
                best_val_acc = val_acc[epoch]
                torch.save(self.model, save_path)
            
            if verbose:
                print(f'Train Acc: {train_acc[epoch]:.4f}, Train Loss: {train_loss[epoch]:>7.6f}', end=', ')
                print(f'Val Acc: {val_acc[epoch]:.4f}, Val Loss: {val_loss[epoch]:>7.6f}', end=' -- ')
                print(f'Training Time: {train_time[epoch]:.2f}s')
        
        self.history = {'train_acc':  train_acc, 
                        'train_loss': train_loss, 
                        'val_acc':    val_acc, 
                        'val_loss':   val_loss, 
                        'time':       train_time}

    def train_epoch(self, train_loader):
        start = time()
        
        self.model.train()
        for data in train_loader:               # Iterate in batches over the training dataset.
            data.to(self.device)                # Train the data if gpu is available
            out = self.model(data)              # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
            
            loss.backward()                     # Derive gradients.
            self.optimizer.step()               # Update parameters based on gradients.
            self.optimizer.zero_grad()          # Clear gradients.
        
        end = time()
        return end - start

    def test(self, loader):
        self.model.eval()

        loss = 0
        correct = 0
        for data in loader:                             # Iterate in batches over the training/test dataset.
            data.to(self.device)                        # Train the data if gpu is available
            out = self.model(data)                      # Predict the outcome by trained model
            pred = out.argmax(dim=1)                    # Use the class with highest probability.

            correct += int((pred == data.y).sum())      # Check against ground-truth labels.
            loss += self.criterion(out, data.y).item()  # Get the loss accumulated of each data sample

        acc = correct / len(loader.dataset)             # Get the accuracy
        avg_loss = loss / len(loader.dataset)           # Get the average loss
        return (acc, avg_loss)                          # Return the accuracy and average loss
    
    def load(self, path):
        self.model = torch.load(path)
        self.model.eval()

    def predict(self, loader):
        preds = []
        with torch.no_grad():
            for data in loader:
                data.to(self.device)
                pred = self.model(data).cpu().detach()
                preds.append(pred)
        preds = torch.vstack(preds)
        return preds