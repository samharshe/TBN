# COPIED FROM EGNN PRO
import torch, wandb, hydra, importlib
from src.utils.data import get_dataloaders
from run.train import train
from run.evaluate import evaluate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # initialize star of show with help of importlib, depending on cfg.name
    model_module = importlib.import_module(f"src.ml.{cfg.name}")
    Model = getattr(model_module, "Model")  # name of relevant class will always be "Model"
    model = Model()
    
    # GPU compatibility
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model) # deprecated; use DDP (but non-trivial); use Torch Lightning
    elif torch.cuda.device_count == 1:
        model = model.to(device)

    # Convert DictConfig to a regular dictionary to make it nice for wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # initialize wandb
    wandb.init(project="TBN", config=cfg_dict)

    # initialize optimizer, scheduler, loss function
    optimizer_class = getattr(torch.optim, cfg.hyperparameters.optimizer.name)
    optimizer = optimizer_class(model.parameters(), 
        lr=cfg.hyperparameters.optimizer.base_learning_rate)
    scheduler_class = getattr(torch.optim.lr_scheduler, cfg.hyperparameters.scheduler.name)
    scheduler = scheduler_class(
        optimizer=optimizer, 
        mode=cfg.hyperparameters.scheduler.mode, 
        factor=cfg.hyperparameters.scheduler.factor, 
        patience=cfg.hyperparameters.scheduler.patience, 
        threshold=cfg.hyperparameters.scheduler.threshold)
    loss_fn_class = getattr(torch.nn, cfg.hyperparameters.training.loss_fn.name)
    loss_fn = loss_fn_class()

    # create dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(version=cfg.data.version,
        train_split=cfg.data.splits.train_split,
        val_split=cfg.data.splits.val_split,
        test_split=cfg.data.splits.test_split,
        batch_size=cfg.hyperparameters.training.batch_size)

    # train and evaluate model
    train(model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          loss_fn=loss_fn,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          max_epochs=cfg.hyperparameters.training.max_epochs,
          early_stop_patience=cfg.hyperparameters.training.early_stop_patience,
          name=cfg.name)
    evaluate(model=model,
        loss_fn=loss_fn,
        test_dataloader=test_dataloader)

    # end wandb run
    wandb.finish()
    
if __name__ == '__main__':
    main()