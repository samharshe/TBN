import torch, sys, functools, wandb
from collections import defaultdict
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Callable
import matplotlib.pyplot as plt

def train_test(model: Module,
               optimizer: Optimizer,
               loss_fn: Callable,
               scheduler: LRScheduler,
               train_dataloader: DataLoader,
               val_dataloader: DataLoader,
               test_dataloader: DataLoader,
               cfg_dict: dict) -> None:
    
    n_epochs = cfg_dict['n_epochs']
    name = cfg_dict['name']
    y_version = cfg_dict['y_version']
    
    wandb.init(project='TBN')
    wandb.config.update(cfg_dict)
    wandb.watch(model, log="all", log_freq=50, log_graph=True)
    try:
        train(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader, n_epochs=n_epochs, name=name)
        if y_version == 'score':
            test_score(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader)
        elif y_version == 'home_win':
            test_classifier(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader)
        else:
            raise ValueError(f'y_version must be either "score" or "home_win". passed value: {y_version}.')
    finally:
        wandb.finish()

def test_classifier(model: Module, 
          loss_fn: Callable, 
          test_dataloader: DataLoader) -> None:
    # track training losses
    test_losses = []
    # track correct predictions
    n_correct, n_total = 0, 0
    # do not track gradients
    model.eval()
    for data in test_dataloader:
        # get input from data item
        x, y = data
        
        # predictions from the model
        y_hat = model(x)

        # keep track of 
        n_correct += torch.sum(torch.sigmoid(y_hat).round() == y).item()
        n_total += len(y)
        
        # loss using passed loss function
        loss = loss_fn(y_hat, y)
        
        # keep track of training losses
        test_losses.append(loss.item())
    
    # calculate and log mean losses from this epoch
    test_mean_loss = torch.mean(torch.tensor(test_losses)).item()
    test_accuracy = n_correct/n_total
    
    # print out results of epoch
    print(f'TEST MEAN LOSS: {test_mean_loss:04f}')
    print(f'TEST ACCURACY: {test_accuracy:04f}')
    
    # Log to wandb
    wandb.log({"test_mean_loss": test_mean_loss, "test_accuracy": test_accuracy})

def test_score(model: Module, 
          loss_fn: Callable, 
          test_dataloader: DataLoader) -> None:
    # track training losses
    test_losses = []
    # track correct predictions
    n_correct, n_total = 0, 0
    # do not track gradients
    model.eval()
    for data in test_dataloader:
        # get input from data item
        x, y = data
        
        # predictions from the model
        y_hat = model(x)
        
        # predicted scores unnormalized given the normalization tensors [175., 176.]
        normalization_tensor = torch.tensor([175., 176.])
        unnormalized_y_hat = y_hat * normalization_tensor
        unnormalized_y = y * normalization_tensor
        home_win_pred = unnormalized_y_hat[:, 0] > unnormalized_y_hat[:, 1]
        home_win_actual = unnormalized_y[:, 0] > unnormalized_y[:, 1]
        
        # keep track of 
        n_correct += torch.sum(home_win_pred == home_win_actual).item()
        n_total += len(y)
        
        # loss using passed loss function
        loss = loss_fn(y_hat, y)
        
        # keep track of training losses
        test_losses.append(loss.item())
    
    # calculate and log mean losses from this epoch
    test_mean_loss = torch.mean(torch.tensor(test_losses)).item()
    test_accuracy = n_correct/n_total
    
    # print out results of epoch
    print(f'TEST MEAN LOSS: {test_mean_loss:04f}')
    print(f'TEST ACCURACY: {test_accuracy:04f}')
    
    # Log to wandb
    wandb.log({"test_mean_loss": test_mean_loss, "test_accuracy": test_accuracy})

def train(model: Module, 
          optimizer: Optimizer, 
          scheduler: LRScheduler, 
          loss_fn: Callable, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader, 
          n_epochs: int, 
          name: str) -> None:
    """trains model on dataloader, saves weights of the best-performing model, and logs ongoing results through wandb.
    
    parameters
    ----------
    model: Module
        self-explanatory.
    optimizer: Optimizer
        self-explanatory.
    scheduler: LRScheduler
        self-explanatory.
    loss_fn: Callable
        self-explanatory.
    train_dataloader: DataLoader
        self-explanatory.
    val_dataloader: DataLoader
        self-explanatory.
    n_epochs: int
        self-explanatory.
    name: str
        best-performing weights are saved to f'/src/weights/{name}.pth'.
    """
    # keep track of the best val performance to know when to save weights
    epoch_val_min_loss = sys.float_info.max
    
    # track training losses
    train_losses = []
    # training loop occurs num_epochs times
    for epoch in range(n_epochs):
        # TRAIN
        train_epoch_losses = []
        
        # track gradients
        model.train()
        
        # iterate through test_dataloader        
        for data in train_dataloader:
            # clear gradients
            optimizer.zero_grad()
            
            # get input from data item
            x, y = data
            
            # predictions from the model
            y_hat = model(x)

            # loss using passed loss function
            loss = loss_fn(y_hat, y)
            
            # calculate gradients
            loss.backward()
            
            # update
            optimizer.step()
            
            # keep track of training losses
            train_epoch_losses.append(loss.item())
            
        epoch_mean_train_loss = torch.mean(torch.tensor(train_epoch_losses)).item()
        wandb.log({"train_mean_loss": epoch_mean_train_loss})
            
        # VAL
        val_epoch_losses = []
        
        # do not track gradients
        model.eval()
        
        # iterate through val_dataloader
        for data in val_dataloader:
            # inputs
            x, y = data
            
            # predictions
            y_hat = model(x)
            
            # canonical loss
            loss = loss_fn(y_hat, y)
            
            # track F_loss, E_loss, canonical loss
            val_epoch_losses.append(loss.item())
        
        # calculate and log mean losses from this epoch
        epoch_mean_val_loss = torch.mean(torch.tensor(val_epoch_losses)).item()
        
        # print out results of epoch
        print(f'EPOCH {epoch+1} OF {n_epochs} | {name} | TRAIN MEAN LOSS: {epoch_mean_train_loss:04f} | VAL MEAN LOSS: {epoch_mean_val_loss:04f}')
        
        # Log to wandb
        wandb.log({
            "val_mean_loss": epoch_mean_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # if this is best val performance yet, save weights
        if epoch_val_min_loss > epoch_mean_val_loss:
            epoch_val_min_loss = epoch_mean_val_loss
            torch.save(model.state_dict(), f'src/weights/{name}.pth')
            
        # update lr based on mean loss of previous epoch
        scheduler.step(epoch_mean_val_loss)
        
        # add epoch losses to training_losses
        train_losses += train_epoch_losses
    
    # plot training loss
    plt.figure()  
    plt.plot(train_losses)
    plt.xlabel('batch #')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.show()
    
def check_memory(model: Module, dataloader: DataLoader, optimizer: Optimizer=torch.optim.SGD, loss_fn: Callable=torch.nn.MSELoss, n_epochs: int=100, batch_size: int=4, lr: float=1e-3) -> None:
    # initialize optimizer and loss function
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_fn = loss_fn()
        
    # make single mini batch
    x, y = next(iter(dataloader))
    x = x[0:batch_size]
    y = y[0:batch_size].squeeze(dim=2)
    
    # track gradients
    model.train()
    # track training losses
    train_losses = []
    # training loop occurs num_epochs times
    for epoch in range(n_epochs):
        # clear gradients
        optimizer.zero_grad()
        
        # predictions from the model
        y_hat = model(x)
        
        # loss using passed loss function
        loss = loss_fn(y_hat, y)
        
        # calculate gradients
        loss.backward()
        
        # update
        optimizer.step()
        
        # keep track of training losses
        train_losses.append(loss.item())
        
    # write gradient and weights and biases stats
    write_stats(model)
    
    # print final prediction
    print(f'final prediction: model(x)')
    print([val for val in y_hat.tolist()])
    print(f'actual value: y')
    print([val for val in y.tolist()])
    print(f'final loss: {train_losses[-1]:0.4f}')

    # plot training losses
    plt.plot(train_losses)
    plt.xlabel('batch #')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.show()
    
def write_stats(model: Module) -> None:
    # write gradient stats
    with open('memorize/gradients.txt', 'w') as f:
        f.write(f"total number of model parameters: {sum(p.numel() for p in model.parameters())}\n" + '-'*60)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                f.write(f"\n{name} | shape: {param.shape}\n")
                f.write(f"gradient stats:\n")
                f.write(f"  norm: {grad.norm():.4f}\n")
                f.write(f"  mean: {grad.mean():.4f}\n")
                f.write(f"  std: {grad.std():.4f}\n")
                f.write(f"  % zeros: {(grad == 0).float().mean()*100:.1f}%\n")
                f.write(f"  % inf/nan: {(~torch.isfinite(grad)).float().mean()*100:.1f}%\n")

    # analyze weight and bias distributions
    with open('memorize/weights_and_biases.txt', 'w') as f:
        f.write("weight and bias distributions\n" + '-'*60)
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                f.write(f"\n{name} | shape: {param.shape}\n")
                f.write(f"  mean: {param.mean().item():.4f}\n")
                f.write(f"  std: {param.std().item():.4f}\n")
                f.write(f"  min: {param.min().item():.4f}\n")
                f.write(f"  max: {param.max().item():.4f}\n")
                f.write(f"  % zeros: {(param == 0).float().mean()*100:.1f}%\n")
