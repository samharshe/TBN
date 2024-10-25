import torch, sys, wandb
from src.ml import model as model_utils
from torch import nn
from torch.utils import data as torch_data
from torch import optim
from torch.optim import lr_scheduler
from typing import Callable
import matplotlib.pyplot as plt

def train_test(model: nn.Module, optimizer: optim.Optimizer, loss_fn: Callable, scheduler: lr_scheduler.LRScheduler, train_dataloader: torch_data.DataLoader, val_dataloader: torch_data.DataLoader, test_dataloader: torch_data.DataLoader, cfg_dict: dict) -> None:
    """trains model on dataloader, saves weights of the best-performing model, and logs ongoing results through wandb.
    
    parameters
    ----------
    model: Module
        self-explanatory.
    optimizer: Optimizer
        self-explanatory.
    loss_fn: Callable
        self-explanatory.
    scheduler: LRScheduler
        self-explanatory.
    train_dataloader: DataLoader
        self-explanatory.
    val_dataloader: DataLoader
        self-explanatory.
    test_dataloader: DataLoader
        self-explanatory.
    cfg_dict: dict
        logged to wandb. used to set n_epochs, name, and y_version.

    returns
    -------
    None
    """
    # initialize variables via cfg_dict
    n_epochs = cfg_dict['n_epochs']
    name = cfg_dict['name'] # used to save weights
    y_version = cfg_dict['y_version'] # used to determine which test function to use
    
    # initialize wandb
    wandb.init(project='TBN') # online training
    # wandb.init(project='TBN' mode='offline') # offline training
    wandb.config.update(cfg_dict)
    wandb.watch(model, log="all", log_freq=50, log_graph=True)

    # train and test model
    try:
        # train model
        train(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, train_dataloader=train_dataloader, val_dataloader=val_dataloader, n_epochs=n_epochs, name=name)
        # test model depending on y_version
        if y_version == 'score':
            test_score(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader)
        elif y_version == 'home_win':
            test_classifier(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader)
        elif y_version == 'player_score':
            test_player_score(model=model, loss_fn=loss_fn, test_dataloader=test_dataloader)
        # perhaps other test functions to be added later
        else:
            raise ValueError(f'y_version must be either "score" or "home_win". passed value: {y_version}.')
    finally:
        wandb.finish()

def test_player_score(model: nn.Module, loss_fn: Callable,  test_dataloader: torch_data.DataLoader) -> None:
    """tests model that predicts player scores.

    parameters
    ----------
    model: Module
        self-explanatory.
    loss_fn: Callable
        self-explanatory.
    test_dataloader: DataLoader
        self-explanatory.
    """
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

        home_y_hat, away_y_hat = model_utils.home_away_tensors(in_tensor=y_hat, original_tensor=x)
        home_y, away_y = model_utils.home_away_tensors(in_tensor=y, original_tensor=x)
        home_y_hat, away_y_hat = torch.sum(home_y_hat, dim=1), torch.sum(away_y_hat, dim=1)
        y_hat = torch.cat((home_y_hat, away_y_hat), dim=1)
        home_y, away_y = torch.sum(home_y, dim=1), torch.sum(away_y, dim=1)
        y = torch.cat((home_y, away_y), dim=1)
        home_win_pred = y_hat[:, 0] > y_hat[:, 1]
        home_win_actual = y[:, 0] > y[:, 1]

        # keep track of correct predictions
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
    
    # log to wandb
    wandb.log({"test_mean_loss": test_mean_loss, "test_accuracy": test_accuracy})

def test_classifier(model: nn.Module, loss_fn: Callable,  test_dataloader: torch_data.DataLoader) -> None:
    """tests model that predicts binary outcomes.

    parameters
    ----------
    model: Module
        self-explanatory.
    loss_fn: Callable
        self-explanatory.
    test_dataloader: DataLoader
        self-explanatory.
    """
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

        # keep track of correct predictions
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
    
    # log to wandb
    wandb.log({"test_mean_loss": test_mean_loss, "test_accuracy": test_accuracy})

def test_score(model: nn.Module, loss_fn: Callable, test_dataloader: torch_data.DataLoader) -> None:
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
        home_win_pred = unnormalized_y_hat[:, 0, 0] > unnormalized_y_hat[:, 1, 0]
        home_win_actual = unnormalized_y[:, 0, 0] > unnormalized_y[:, 1, 0]
        
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
    # wandb.log({"test_mean_loss": test_mean_loss, "test_accuracy": test_accuracy})

def train(model: nn.Module, optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, loss_fn: Callable, train_dataloader: torch_data.DataLoader, val_dataloader: torch_data.DataLoader, n_epochs: int, name: str) -> None:
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

    returns
    -------
    None; logs results to wandb and prints epoch-by-epoch mean train and val losses.
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
            
        # calculate and log mean losses from this epoch
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
        
        # log to wandb
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
    
def check_memory(model: nn.Module, dataloader: torch_data.DataLoader, optimizer: optim.Optimizer = optim.SGD, loss_fn: Callable = nn.MSELoss, n_epochs: int = 100, batch_size: int = 4, lr: float = 1e-3) -> None:
    """checks memory usage of model.
    
    parameters
    ----------
    model: Module
        self-explanatory.
    dataloader: DataLoader
        self-explanatory.

    returns
    -------
    None; prints final prediction, actual value, and final loss and makes matplotlib plot of training losses.
    """
    try:
        # make single mini batch
        x, y = next(iter(dataloader))
        x = x[0:batch_size]
        y = y[0:batch_size]
    except ValueError:
        raise ValueError(f'batch_size passed into this function must be an integer less than or equal to the batch size of the dataloader passed into this function. batch_size passed into this function: {batch_size}. batch size of dataloader passed into this function: {next(iter(dataloader))[0].shape[0]}.')

    # initialize optimizer and loss function
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_fn = loss_fn()
    
    # track gradients
    model.train()
    # track training losses
    train_losses = []
    # training loop occurs num_epochs times
    for _ in range(n_epochs):
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
        
    # write gradient and weights and biases stats to memorize/
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
    
def write_stats(model: nn.Module) -> None:
    """writes gradient stats to memorize/gradients.txt and weight and bias stats to memorize/weights_and_biases.txt.
    
    parameters
    ----------
    model: Module
        self-explanatory.
    
    returns
    -------
    None; modifies memorize/gradients.txt and memorize/weights_and_biases.txt.
    """
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