import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from glob import glob
import torch
# import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
import numpy as np
from models.epsnet import get_model
from models.gradient_penalty import get_lambda_gp_schedule
from utils.datasets import ConformationDataset, TSDataset
from utils.transforms import CountNodesPerGraph
from utils.misc import (
    seed_all,
    get_new_log_dir,
    get_logger,
    get_checkpoint_path,
    inf_iterator,
)
from utils.common import get_optimizer, get_scheduler, get_scaler_optimizer, get_main_optimizer


import wandb

if __name__ == "__main__":
    # TODO: remove parallel_training entry

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=False)
    parser.add_argument("--resume_iter", type=int, default=None)
    parser.add_argument("--resume_config", type=int, default=None) 
    args = parser.parse_args()

    resume = args.resume
    if resume:
        config_path = glob(os.path.join(args.resume_config, "*.yml"))[0]
        resume_from = args.resume_config
    else:
        config_path = os.path.join(os.getcwd(), "configs/train_config.yaml")

    with open(config_path, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    device = config.configuration.device
    logdir = config.configuration.logdir
    pretrain = config.configuration.pretrain
    project = config.configuration.project
    name = config.configuration.name
    tag = config.configuration.tag
    fn = config.configuration.fn

    torch.randn(1).to(device)

    config_name = os.path.basename(config_path)[
        : os.path.basename(config_path).rfind(".")
    ]
    seed_all(config.train.seed)

    if tag is None:
        tag = name

    # Logging
    if resume:
        log_dir = get_new_log_dir(
            logdir, prefix=config_name, tag=f"{tag}_resume", fn=fn
        )
        os.symlink(
            os.path.realpath(resume_from),
            os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))),
        )
    else:
        log_dir = get_new_log_dir(
            logdir, prefix=config_name, tag=f"{tag}", fn=fn
        )
        shutil.copytree("./models", os.path.join(log_dir, "models"))

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger("train", log_dir)
    logger.info(config)

    use_wandb = False
    if name and project:
        use_wandb = True
        wandb.init(project=project, name=name)
        wandb.config = config

    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    logger.info("Loading datasets...")
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    train_iterator = inf_iterator(
        DataLoader(train_set, config.train.batch_size, shuffle=True)
    )
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False)

    # Model
    # TODO: remove parallel_training entry
    logger.info("Building model...")
    config.model.parallel_training = config.parallel_training
    model = get_model(config.model).to(device)
    # print(model)
    
    # Optimizer
    optimizer = get_main_optimizer(config.train.optimizer, model) if config.parallel_training.enable else get_optimizer(config.train.optimizer, model)
    if config.parallel_training.enable:
        scaler_optimizer = get_scaler_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    start_iter = 1

    lambda_gp_schedule = get_lambda_gp_schedule(
        schedule=config.train.gradient_penalty,
        total_time=config.train.max_iters,
    )
    
    # Resume from checkpoint
    if resume:
        ckpt_path, start_iter = get_checkpoint_path(
            os.path.join(resume_from, "checkpoints"), it=args.resume_iter
        )
        logger.info("Resuming from: %s" % ckpt_path)
        logger.info("Iteration: %d" % start_iter)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    if pretrain:
        logger.info(f"pretraining model checkpoint load : {pretrain}")
        ckpt = torch.load(pretrain, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    def train(it, mode, accumulation_steps=4):
        """
        Train function with gradient accumulation over 4 steps.

        Args:
            it (int): Current iteration number.
            mode (str): Training mode ('scaler', 'edge', etc.).
            accumulation_steps (int): Number of steps to accumulate gradients (default: 4).
        """
        model.train()
        optimizer.zero_grad()  # Reset gradients at the start of accumulation
        total_loss_sum = 0.0   # Accumulate loss sum over steps
        total_n = 0            # Accumulate number of samples
        orig_grad_norm = 0.0   # Store gradient norm after accumulation

        for step in range(accumulation_steps):
            # Get batch and move to device
            batch = next(train_iterator).to(device)
            
            # Handle scaler mode
            if mode == 'scaler':
                property_name = config.parallel_training.name
                prop_value = getattr(batch, property_name, None)
                prop_value = prop_value.clone().detach().to(device).float()
                if step == 0:  # Only zero scaler_optimizer once at the start
                    scaler_optimizer.zero_grad()
            else:
                prop_value = None

            lambda_gp = lambda_gp_schedule(it)

            # Freeze/unfreeze parameters based on mode
            if mode != "edge":
                for name, param in model.named_parameters():
                    if "scaler_mlp" not in name:
                        param.requires_grad = False
            else:  # Unfreeze all parameters
                for param in model.parameters():
                    param.requires_grad = True
                    model.betas.requires_grad = False
                    model.alphas.requires_grad = False

            # Compute loss
            loss = model.get_loss(
                atom_type=batch.atom_type,
                feat=batch.feat,
                pos=batch.pos,
                aromatic_rings=batch.aromatic_rings,
                force_planarity=config.train.force_planarity,
                bond_index=batch.edge_index,
                bond_type=batch.edge_type,
                batch=batch.batch,
                num_nodes_per_graph=batch.num_nodes_per_graph,
                num_graphs=batch.num_graphs,
                lambda_gp=lambda_gp,
                sinkhorn_knopp=config.train.sinkhorn_knopp,
                mode=mode,
                prop=prop_value,
                hydrogen_weight_method=config.train.hydrogen_weight,
            )

            # Accumulate loss statistics
            n = loss.size(0)
            total_loss_sum += loss.sum().item()  # Accumulate sum for reporting
            total_n += n                         # Accumulate sample count
            loss = loss.mean()                   # Mean loss for backprop

            # Backpropagate and accumulate gradients
            loss.backward()  # Gradients are accumulated in model parameters

            # Only update parameters and compute gradient norm after last step
            if step == accumulation_steps - 1:
                if lambda_gp == 0.0:
                    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                else:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.norm(2).item() ** 2
                    orig_grad_norm = total_norm ** 0.5
                    
                    if hasattr(lambda_gp_schedule, "gradient_history"):
                        lambda_gp_schedule.gradient_history.append(abs(orig_grad_norm))

                # Perform optimization step
                if mode == 'scaler':
                    scaler_optimizer.step()
                    scaler_optimizer.zero_grad()  # Reset after step
                else:
                    optimizer.step()
                    optimizer.zero_grad()  # Reset after step

        return (
            total_loss_sum,           # Total accumulated loss sum
            optimizer.param_groups[0]["lr"],  # Learning rate
            orig_grad_norm,           # Gradient norm after accumulation
            total_n                   # Total number of samples
        )

    def validate(it, mode):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_loader):
                batch = batch.to(device)
    
                if mode == 'scaler':
                    property_name = config.parallel_training.name
                    prop_value = getattr(batch, property_name, None)
                    prop_value = prop_value.clone().detach().to(device).float()
                else:
                    prop_value=None

                loss = model.get_loss(
                    atom_type=batch.atom_type,
                    feat=batch.feat,
                    pos=batch.pos,
                    aromatic_rings=batch.aromatic_rings,
                    force_planarity=config.train.force_planarity,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    lambda_gp = 0,
                    sinkhorn_knopp=config.train.sinkhorn_knopp,
                    mode = mode,
                    prop = prop_value,
                    hydrogen_weight_method=config.train.hydrogen_weight, 
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
        avg_loss = sum_loss / sum_n

        if config.train.scheduler.type == "plateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        logger.info("[Validate] Mode %s | Iter %05d | Loss %.6f " % (mode, it, avg_loss))
        if use_wandb:
            wandb.log({"val/loss": avg_loss, })
        return avg_loss

    try:
        logger.info("Training...")
        if config.parallel_training.enable:
            time1 = config.parallel_training.train.time_edge  # train edge network for time1 iterations
            time2 = config.parallel_training.train.time_scaler  # train scaler network for time2 iterations
            mode = "edge"  # Start with edge training
            edge_counter, scaler_counter = 0, 0
        else:
            mode = "edge"
        
        loss_sum = 0
        n_sum = 0
        grad_norm_sum = 0
        best_loss = 10000
        for it in range(start_iter, config.train.max_iters + 1):
            # Update counters and switch mode if needed
            if config.parallel_training.enable:
                if mode == "edge":
                    edge_counter += 1
                    if edge_counter > time1:
                        mode = "scaler"
                        edge_counter = 0
                elif mode == "scaler":
                    scaler_counter += 1
                    if scaler_counter > time2:
                        mode = "edge"
                        scaler_counter = 0
            else:
                mode="edge"
            
            loss, lr, grad_norm, n = train(it, mode, accumulation_steps=config.train.accumulation_steps)
            loss_sum += loss
            n_sum += n
            grad_norm_sum += grad_norm
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss_sum / n_sum,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm_sum / config.train.val_freq,
                        }
                    )
                logger.info(
                    "[Train] Mode %s | Iter %05d | Loss %.2f | Grad %.2f | LR %.6f"
                    % ( 
                        mode,
                        it*int(config.train.accumulation_steps),
                        loss_sum / n_sum,
                        grad_norm_sum / config.train.val_freq,
                        lr,
                    )
                )
                loss_sum = 0
                n_sum = 0
                grad_norm_sum = 0
                avg_val_loss = validate(it, mode)
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)

                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "iteration": it,
                            "avg_val_loss": avg_val_loss,
                        },
                        ckpt_path,
                    )

    except KeyboardInterrupt:
        logger.info("Terminating...")
