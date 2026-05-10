import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset
import tqdm

import utils.compat
from utils.add_ccn_noise import *
from utils.cifar_data_utils import Custom_dataset, Double_dataset
from utils.directional_diffusion_model import *
from utils.ema import EMA
from utils.learning import *
from utils.log_config import setup_logger
from utils.pre_correction import *
from bbbc021_dataset import BBBC021Dataset


# Simple feature encoder for pre-extracted features
class FeatureEncoder(nn.Module):
    """Simple linear encoder for pre-extracted features."""
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, input_dim)
        return self.fc(x)


# Dataset wrapper for BBBC021 with weak/strong feature views
class FeatureDoubleDataset(Dataset):
    """
    Dataset wrapper that creates weak and strong views from pre-extracted features.
    For features, weak/strong just means the same features (identity).
    """
    def __init__(self, dataset, feature_dim, fp_dim=512):
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.fp_dim = fp_dim
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        features, targets, mask = self.dataset[idx]
        # For features, weak and strong are the same
        # features shape: (feature_dim,) - DO NOT add batch dim, DataLoader will handle it
        return features, features, targets, idx


# Dataset wrapper for BBBC021 test set
class FeatureCustomDataset(Dataset):
    """
    Dataset wrapper for test set with pre-extracted features.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        features, targets, mask = self.dataset[idx]
        # features shape: (feature_dim,) - DO NOT add batch dim, DataLoader will handle it
        return features, targets, idx


# Main training function
def train(
    diffusion_model, train_dataset, test_dataset, model_path, args, feature_dim
):
    """
    Train the diffusion model with the given datasets and arguments.
    """
    print(
        f"Use loss weights: {args.loss_w}, Use Single label: {args.to_single_label}, Use One view: {args.one_view}"
    )
    
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    num_models = diffusion_model.num_models
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    noise_class = args.noise_type.split("-")[1]
    noise_ratio = float(args.noise_type.split("-")[2])

    # Handle noisy labels if needed
    if noise_class == "idn":
        if "bbbc021" in args.noise_type and noise_ratio == 0.0:
            print("Training on pure label:", args.noise_type)
        else:
            print(f"IDN noise not implemented for BBBC021, using clean labels")
        noisy_labels = torch.tensor(train_dataset.dataset.targets_data).to(device)
    elif noise_class == "sym":
        if "bbbc021" in args.noise_type and noise_ratio == 0.0:
            print("Training on pure label:", args.noise_type)
        else:
            print(f"Symmetric noise not implemented for BBBC021, using clean labels")
        noisy_labels = torch.tensor(train_dataset.dataset.targets_data).to(device)
    elif noise_class == "asym":
        if "bbbc021" in args.noise_type and noise_ratio == 0.0:
            print("Training on pure label:", args.noise_type)
        else:
            print(f"Asymmetric noise not implemented for BBBC021, using clean labels")
        noisy_labels = torch.tensor(train_dataset.dataset.targets_data).to(device)
    else:
        print("Check your noise type carefully!")
        noisy_labels = torch.tensor(train_dataset.dataset.targets_data).to(device)

    # Compute embedding fp(x) for dataset
    # For pre-extracted features, we compute fp(x) using the feature encoder
    print("Computing fp embeddings for dataset")
    weak_embed, strong_embed = prepare_2_fp_x_feature(
        diffusion_model.fp_encoder,
        train_dataset,
        device=device,
        fp_dim=args.feature_dim,
    )
    weak_embed = weak_embed.to(device)
    strong_embed = strong_embed.to(device)

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=200, shuffle=False, num_workers=args.num_workers
    )

    # Optimizer settings
    if diffusion_model.num_models == 1:
        optimizer = optim.Adam(
            diffusion_model.model.parameters(),
            lr=0.0001,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=1e-08,
        )
        ema_helper = EMA(mu=0.999)
        ema_helper.register(diffusion_model.model)
    elif diffusion_model.num_models == 2:
        optimizer_res = optim.Adam(
            diffusion_model.model0.parameters(),
            lr=0.0001,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=1e-08,
        )
        optimizer_noise = optim.Adam(
            diffusion_model.model1.parameters(),
            lr=0.0001,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=False,
            eps=1e-08,
        )
        ema_helper_res = EMA(mu=0.999)
        ema_helper_noise = EMA(mu=0.999)
        ema_helper_res.register(diffusion_model.model0)
        ema_helper_noise.register(diffusion_model.model1)

    diffusion_loss = nn.MSELoss(reduction="none")

    # Train in a loop
    max_accuracy = 0.0
    print("Directional Diffusion training start")
    for epoch in range(n_epochs):
        if diffusion_model.num_models == 1:
            diffusion_model.model.train()
        else:
            diffusion_model.model0.train()
            diffusion_model.model1.train()
        total_loss = 0.0
        total_batches = 0

        with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train diffusion epoch {epoch}",
            ncols=120,
        ) as pbar:
            for i, data_batch in pbar:
                [x_batch_w, x_batch_s, y_batch, data_indices] = data_batch[:4]
                x_batch_w = x_batch_w.to(device)
                x_batch_s = x_batch_s.to(device)
                y_noisy = y_batch.to(device)

                # For features, use the feature encoder directly
                fp_embd_w = diffusion_model.fp_encoder(x_batch_w.to(device))
                fp_embd_s = diffusion_model.fp_encoder(x_batch_s.to(device))

                # pre-correct labels based on two views
                (
                    y_label_batch_w,
                    y_label_batch_s,
                    loss_weights_w,
                    loss_weights_s,
                    y_label_batch_n,
                    gamma_batch,
                ) = precorrect_labels_in_two_view(
                    fp_embd_w=fp_embd_w,
                    fp_embd_s=fp_embd_s,
                    y_noisy=y_noisy,
                    weak_embed=weak_embed,
                    strong_embed=strong_embed,
                    noisy_labels=noisy_labels,
                    k=k,
                    n_class=n_class,
                    use_cosine_similarity=args.use_cos,
                    to_single_label=args.to_single_label,
                )

                if args.one_view:
                    x_batch = x_batch_w
                else:
                    x_batch = (
                        1 - gamma_batch.view(-1, 1)
                    ) * x_batch_w + gamma_batch.view(-1, 1) * x_batch_s

                # Check if the labels are one-hot encoded
                if len(y_label_batch_w.shape) == 1:
                    y_one_hot_batch_w = cast_label_to_one_hot_and_prototype(
                        y_label_batch_w.to(torch.int64), n_class=n_class
                    )
                    y_one_hot_batch_s = cast_label_to_one_hot_and_prototype(
                        y_label_batch_s.to(torch.int64), n_class=n_class
                    )
                else:
                    y_one_hot_batch_w = y_label_batch_w
                    y_one_hot_batch_s = y_label_batch_s

                y_0_batch_w = y_one_hot_batch_w.to(device)
                y_0_batch_s = y_one_hot_batch_s.to(device)
                y_zeros = torch.zeros_like(y_0_batch_w)
                y_n_batch = y_label_batch_n.to(device)

                # Adjust learning rate
                if diffusion_model.num_models == 1:
                    adjust_learning_rate(
                        optimizer,
                        i / len(train_loader) + epoch,
                        warmup_epochs=warmup_epochs,
                        n_epochs=n_epochs,
                        lr_input=1e-3,
                    )
                else:
                    adjust_learning_rate(
                        optimizer_res,
                        i / len(train_loader) + epoch,
                        warmup_epochs=warmup_epochs,
                        n_epochs=n_epochs,
                        lr_input=1e-3,
                    )
                    adjust_learning_rate(
                        optimizer_noise,
                        i / len(train_loader) + epoch,
                        warmup_epochs=warmup_epochs,
                        n_epochs=n_epochs,
                        lr_input=1e-3,
                    )

                # Sampling t
                n = x_batch.size(0)
                t = torch.randint(
                    low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)
                ).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # Forward pass
                output, e = diffusion_model.forward_t(
                    y_zeros, y_0_batch_w, x_batch, t, fp_embd_w
                )

                if diffusion_model.objective == "pred_res_noise":
                    L_res = diffusion_loss(output[0], e[0])
                    L_noise = diffusion_loss(output[1], e[1])

                    weighted_L_res = (
                        torch.matmul(loss_weights_w, L_res) if args.loss_w else L_res
                    )
                    weighted_L_noise = (
                        torch.matmul(loss_weights_w, L_noise)
                        if args.loss_w
                        else L_noise
                    )

                    l_res_loss = torch.mean(weighted_L_res)
                    l_noise_loss = torch.mean(weighted_L_noise)

                    if diffusion_model.num_models == 2:
                        optimizer_res.zero_grad()
                        l_res_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            diffusion_model.model0.parameters(), 1.0
                        )
                        optimizer_res.step()
                        ema_helper_res.update(diffusion_model.model0)

                        optimizer_noise.zero_grad()
                        l_noise_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            diffusion_model.model1.parameters(), 1.0
                        )
                        optimizer_noise.step()
                        ema_helper_noise.update(diffusion_model.model1)

                        pbar.set_postfix(
                            {
                                "res_loss": l_res_loss.item(),
                                "noise_loss": l_noise_loss.item(),
                            }
                        )

                    elif diffusion_model.num_models == 1:
                        loss = 0.1 * l_res_loss + 0.9 * l_noise_loss
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            diffusion_model.model.parameters(), 1.0
                        )
                        optimizer.step()
                        ema_helper.update(diffusion_model.model)
                        pbar.set_postfix({"loss": loss.item()})

                else:
                    L_noise = diffusion_loss(output, e)
                    weighted_L_noise = (
                        torch.matmul(loss_weights_w, L_noise)
                        if args.loss_w
                        else L_noise
                    )
                    loss = torch.mean(weighted_L_noise)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        diffusion_model.model.parameters(), 1.0
                    )
                    optimizer.step()
                    ema_helper.update(diffusion_model.model)
                    pbar.set_postfix({"loss": loss.item()})

        # Validation and model saving
        if epoch >= warmup_epochs:
            test_acc = test(diffusion_model, test_loader, feature_dim)
            logger.info(f"epoch: {epoch}, test accuracy: {test_acc:.2f}%")
            if test_acc > max_accuracy:
                print("Improved! Evaluate on testing set...")
                if diffusion_model.num_models == 1:
                    states = {
                        "model": diffusion_model.model.state_dict(),
                        "fp_encoder": diffusion_model.fp_encoder.state_dict(),
                    }
                else:
                    states = {
                        "model0": diffusion_model.model0.state_dict(),
                        "model1": diffusion_model.model1.state_dict(),
                        "fp_encoder": diffusion_model.fp_encoder.state_dict(),
                    }
                torch.save(states, model_path)
                message = f"Model saved, update best accuracy at Epoch {epoch}, test acc: {test_acc}"
                logger.info(message)
                max_accuracy = max(max_accuracy, test_acc)


def test(diffusion_model, test_loader, feature_dim):
    """
    Test the diffusion model.
    """
    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0
        all_cnt = 0
        for idx, data_batch in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f"Doing DDIM...",
            ncols=100,
        ):
            [images, target, _] = data_batch[:3]
            target = target.to(diffusion_model.device)
            # Reshape images to (batch, feature_dim)
            images = images.to(diffusion_model.device)
            if len(images.shape) == 3:
                images = images.squeeze(1)  # Remove extra dimension
            
            label_t_0 = diffusion_model.ddim_sample(
                x_batch=images, y_input=0, fp_x=None, last=True, stochastic=False
            )
            correct = cnt_agree(label_t_0.detach(), target)
            correct_cnt += correct
            all_cnt += images.shape[0]

    acc = 100 * correct_cnt / all_cnt
    return acc


def prepare_2_fp_x_feature(fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400):
    """
    Prepare feature embeddings for pre-extracted features.
    """
    # Initialize feature embeddings
    fp_embed_all_weak = torch.zeros([len(dataset), fp_dim], device=device)
    fp_embed_all_strong = torch.zeros([len(dataset), fp_dim], device=device)

    with torch.no_grad():
        data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing embeddings fp(x)', ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch_weak, x_batch_strong, _, data_indices] = data_batch[:4]
                temp_weak = fp_encoder(x_batch_weak.to(device))
                temp_strong = fp_encoder(x_batch_strong.to(device))
                data_indices = data_indices.to(device)
                fp_embed_all_weak[data_indices] = temp_weak
                fp_embed_all_strong[data_indices] = temp_strong

    return fp_embed_all_weak.cpu(), fp_embed_all_strong.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    # Training parameters
    parser.add_argument(
        "--noise_type", default="bbbc021-sym-0.0", help="noise label file", type=str
    )
    parser.add_argument(
        "--nepoch", default=200, help="number of training epochs", type=int
    )
    parser.add_argument("--batch_size", default=64, help="batch_size", type=int)
    parser.add_argument("--device", default="cuda:0", help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=3, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    # Diffusion model hyperparameters
    parser.add_argument("--num_models", default=2, help="number of models", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn or cos", type=int)
    parser.add_argument(
        "--loss_w", default=True, help="use weights for loss", action="store_false"
    )
    parser.add_argument(
        "--to_single_label",
        default=False,
        help="use single_label for label sampling",
        action="store_true",
    )
    parser.add_argument(
        "--one_view", default=False, help="use single view", action="store_true"
    )
    parser.add_argument("--use_cos", default=True, help="use cos", action="store_false")
    parser.add_argument(
        "--ddim_n_step", default=10, help="number of steps in ddim", type=int
    )
    parser.add_argument(
        "--diff_encoder",
        default="resnet34",
        help="which encoder for diffusion (linear, resnet18, 34, 50...)",
        type=str,
    )
    parser.add_argument(
        "--objective",
        default="pred_res_noise",
        help="which type for diffusion (pred_res, pred_noise, pred_res_noise...)",
        type=str,
    )
    # Feature encoder parameters
    parser.add_argument(
        "--fp_encoder",
        default="FeatureEncoder",
        help="which encoder for fp (FeatureEncoder for pre-extracted features)",
        type=str,
    )
    # Storage path
    parser.add_argument(
        "--log_name",
        default="bbbc021-sym-0.0.log",
        help="create your logs name",
        type=str,
    )
    # Dataset path
    parser.add_argument(
        "--data_file",
        default="_data/BBBC021/BBBC021_dataset_complete_one_fold.csv",
        help="path to BBBC021 dataset CSV file",
        type=str,
    )
    args = parser.parse_args()
    logger = setup_logger(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set GPU or CPU for training
    device = args.device
    print(device)
    print("Using device:", device)

    # Load BBBC021 dataset
    print("Loading BBBC021 dataset...")
    train_dataset_raw = BBBC021Dataset(
        data_file=args.data_file,
        split="train",
        mask_uncertain=True,
    )
    test_dataset_raw = BBBC021Dataset(
        data_file=args.data_file,
        split="test",
        mask_uncertain=True,
    )

    n_class = len(train_dataset_raw.get_target_names())
    feature_dim = len(train_dataset_raw.get_feature_names())
    
    print(f"Number of classes (MoA): {n_class}")
    print(f"Number of features: {feature_dim}")
    print(f"Training samples: {len(train_dataset_raw)}")
    print(f"Test samples: {len(test_dataset_raw)}")

    # Create feature encoder
    fp_encoder = FeatureEncoder(input_dim=feature_dim, output_dim=args.feature_dim).to(device)
    fp_dim = args.feature_dim

    # Wrap datasets for training
    train_dataset = FeatureDoubleDataset(train_dataset_raw, feature_dim, fp_dim)
    test_dataset = FeatureCustomDataset(test_dataset_raw)

    # Initialize the diffusion model
    model_path = f"./model/DLD_{args.fp_encoder}_{args.noise_type}.pt"
    base_model = DirectionalConditionalModel(
        n_steps=1000,
        y_dim=n_class,
        fp_dim=fp_dim,
        feature_dim=args.feature_dim,
        guidance=True,
        num_models=args.num_models,
        objective=args.objective,
        encoder_type=args.diff_encoder,
        use_feature_input=True,  # Skip diffusion_encoder for pre-extracted features
    ).to(device)

    diffusion_model = DirectionalDiffusion(
        model=base_model,
        fp_encoder=fp_encoder,
        num_models=args.num_models,
        num_timesteps=1000,
        n_class=n_class,
        fp_dim=fp_dim,
        device=device,
        feature_dim=args.feature_dim,
        encoder_type=args.diff_encoder,
        objective=args.objective,
        sampling_timesteps=args.ddim_n_step,
        condition=True,
        convert_to_ddim=False,
        sum_scale=1.0,
        ddim_sampling_eta=0.0,
        beta_schedule="cosine",
        use_feature_input=True,  # Skip diffusion_encoder for pre-extracted features
    )

    diffusion_model.fp_encoder.eval()

    # Train the diffusion model
    print(f"Training DLD using fp encoder: {args.fp_encoder} on: {args.noise_type}.")
    print(f"Model saving dir: {model_path}")
    train(
        diffusion_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_path=model_path,
        args=args,
        feature_dim=feature_dim,
    )
