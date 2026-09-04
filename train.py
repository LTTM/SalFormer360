import os
import time
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VRVideo, get_dataset_config
from data_transformation import image_transform
from model import TemporalSegFormer
from losses import CC_Value, KL_loss, sphere_mse, BCE_L

# ============================================================
# Configuration
# ============================================================
DATASET = "eye"  # "360", "eye", or "pvs"

BATCH_SIZE = 4
NUM_WORKERS = 6
EPOCHS = 20

TOTAL_ITERS = {
    "360": 14600,
    "pvs": 1100,
    "eye": 6300,
}

PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 1

# Optional overrides. Leave as None to use dataset.py defaults.
DATA_ROOT = None
HEAD_DATA = None
IFCB_IMAGE = None

SAVE_DIR = "./results"



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weights_init_he(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def train_step(last_sal, gth_sal, color_image, model, BATCH_SIZE, device, optim_model, vid, fid, ifcb_image):
    model.train()
    g_image = model(color_image, ifcb_image, fid)
    cc_loss = (1 - CC_Value(g_image, gth_sal)).to(device)
    kl_loss = KL_loss(g_image, gth_sal).to(device)
    s_mse_loss = sphere_mse(g_image, gth_sal, 224, 384).to(device)
    bce_loss = BCE_L(g_image, gth_sal).to(device)
    tot_loss = (kl_loss) + (cc_loss) + (s_mse_loss) + (bce_loss)
    optim_model.zero_grad()
    tot_loss.backward()
    optim_model.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return tot_loss, kl_loss, cc_loss, s_mse_loss, bce_loss


def main():
    global DATASET, DATA_ROOT, HEAD_DATA, IFCB_IMAGE, SAVE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["360", "eye", "pvs"], default=DATASET)
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--metadata", default=HEAD_DATA)
    parser.add_argument("--bias", default=IFCB_IMAGE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    total_iters = TOTAL_ITERS[args.dataset]
    root = args.root or config["root"]
    metadata = args.metadata or config["pickle_file"]
    bias = args.bias or config["bias_file"]
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Total training iterations: {total_iters}", flush=True)
    print(f"Root: {root}", flush=True)

    train_dataset = VRVideo(
        root, 224, 384,
        frame_interval=5, train=True, cache_gt=True,
        transform=image_transform, pickle_file_path=metadata,
        bias_file_path=bias, dataset_name=args.dataset,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(0), pin_memory=True,
    )

    model = TemporalSegFormer(PRETRAINED_MODEL, NUM_CLASSES)
    model.decoder.apply(weights_init_he)
    model.to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 0.0001, "weight_decay": 1e-5},
            {"params": model.decoder.parameters(), "lr": 0.001, "weight_decay": 1e-5},
            {"params": [model.alpha], "lr": 0.1},
            {"params": [model.beta], "lr": 0.0001},
        ]
    )


    g_losses = []
    g_cc_losses = []
    g_kl_losses = []
    g_s_mse_losses = []
    g_bce_losses = []

    writer = SummaryWriter()
    previous_epochs = 0
    iteration = 0
    stop_training = False

    print("Training model started", flush=True)

    for epoch in range(EPOCHS):
        start = time.time()
        losses = []
        cc_losses = []
        kl_losses = []
        s_mse_losses = []
        bce_losses = []

        for colored_image, last_sal, gth_sal, vid, fid, bias in train_dataloader:
            if iteration > total_iters:
                stop_training = True
                break

            colored_image = torch.stack(colored_image, dim=1).to(device)
            last_sal = last_sal.to(device)
            gth_sal = gth_sal.to(device)

            loss, kl_loss, cc_loss, s_mse_loss, bce_loss = train_step(
                last_sal, gth_sal, colored_image, model, BATCH_SIZE,
                device, optimizer, vid, fid, bias,
            )
            iteration += 1

            losses.append(loss.detach().cpu())
            cc_losses.append(cc_loss.detach().cpu())
            kl_losses.append(kl_loss.detach().cpu())
            s_mse_losses.append(s_mse_loss.detach().cpu())
            bce_losses.append(bce_loss.detach().cpu())

            if iteration == total_iters:
                file_path = os.path.join(save_dir, f"model_params_{previous_epochs + epoch + 1}_{iteration}.pth")
                torch.save(model.state_dict(), file_path)
                print(f"\n\t Network parameters saved to {file_path}", flush=True)

                file_path = os.path.join(save_dir, f"optim_{previous_epochs + epoch + 1}_{iteration}.pth")
                torch.save(optimizer.state_dict(), file_path)
                print(f"\t Optimizer saved to {file_path}", flush=True)

                data = {
                    "g_losses": g_losses,
                    "g_cc_losses": g_cc_losses,
                    "g_kl_losses": g_kl_losses,
                    "g_s_mse_losses": g_s_mse_losses,
                    "g_bce_losses": g_bce_losses,
                }
                file_path = os.path.join(save_dir, f"Training_losses_{previous_epochs + epoch + 1}_{iteration}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
                print(f"\t Losses saved to {file_path} \n", flush=True)

        if stop_training:
            break

        g_losses.append(np.mean(losses))
        g_cc_losses.append(np.mean(cc_losses))
        g_kl_losses.append(np.mean(kl_losses))
        g_s_mse_losses.append(np.mean(s_mse_losses))
        g_bce_losses.append(np.mean(bce_losses))

        writer.add_scalar("Training_Loss/train_tot_loss", np.mean(losses), epoch)
        writer.add_scalar("Training_Loss/train_cc_losses", np.mean(cc_losses), epoch)
        writer.add_scalar("Training_Loss/train_kl_losses", np.mean(kl_losses), epoch)
        writer.add_scalar("Training_Loss/train_s_mse_losses", np.mean(s_mse_losses), epoch)
        writer.add_scalar("Training_Loss/train_bce_losses", np.mean(bce_losses), epoch)
        writer.add_scalar("Learning Rate/encoder", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Learning Rate/decoder", optimizer.param_groups[1]["lr"], epoch)
        writer.add_scalar("Learning Rate/alpha", optimizer.param_groups[2]["lr"], epoch)

        print(f"\n Time for epoch {previous_epochs + epoch + 1} is {time.time() - start} sec", flush=True)
        print(f"\n\t partial train loss for the model (single batch): {np.mean(losses):f}", flush=True)
        print(f"\t partial train loss for cc_losses (single batch): {np.mean(cc_losses):f}", flush=True)
        print(f"\t partial train loss for kl_losses (single batch): {np.mean(kl_losses):f}", flush=True)
        print(f"\t partial train loss for s_mse_losses (single batch): {np.mean(s_mse_losses):f}", flush=True)
        print(f"\t partial train loss for bce_losses (single batch): {np.mean(bce_losses):f}", flush=True)

    writer.close()
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
