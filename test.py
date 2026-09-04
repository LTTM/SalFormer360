import os
import time
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VRVideo, get_dataset_config
from data_transformation import test_transform
from model import TemporalSegFormer
from losses import CC_Value, KL_loss, sphere_mse, BCE_L
from metrics import evaluation_metrics, load_data

# ============================================================
# Configuration
# ============================================================
DATASET = "eye"  # "360", "eye", or "pvs"

MODEL_FILES = {
    "360": "model_params_6_14600.pth",
    "pvs": "model_params_1_1100.pth",
    "eye": "model_params_1_6300.pth",
}

BATCH_SIZE = 4
NUM_WORKERS = 6
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 1

# Optional overrides. Leave as None to use dataset.py defaults.
DATA_ROOT = None
HEAD_DATA = None
IFCB_IMAGE = None
SAVE_DIR = "./results"
MODEL_WEIGHTS_DIR = "./model_weights"



def test_epoch(last_sal, gth_sal, color_image, model, BATCH_SIZE, device, vid, fid, ifcb_image):
    model.eval()
    with torch.no_grad():
        g_image = model(color_image, ifcb_image, fid)
        cc_loss = 1 - CC_Value(g_image, gth_sal)
        kl_loss = KL_loss(g_image, gth_sal)
        s_mse_loss = sphere_mse(g_image, gth_sal, 224, 384)
        bce_loss = BCE_L(g_image, gth_sal)
        tot_loss = cc_loss + kl_loss + s_mse_loss + bce_loss
    return tot_loss, kl_loss, cc_loss, s_mse_loss, bce_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["360", "eye", "pvs"], default=DATASET)
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--metadata", default=HEAD_DATA)
    parser.add_argument("--bias", default=IFCB_IMAGE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    
    #parser.add_argument("--model", default=MODEL_FILE)
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    root = args.root or config["root"]
    metadata = args.metadata or config["pickle_file"]
    bias = args.bias or config["bias_file"]
    model_file = MODEL_FILES[args.dataset]
    save_dir = args.save_dir
    #model_file = args.model
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Selected device: {device}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Loading model: {model_file}", flush=True)
    print(f"Root: {root}", flush=True)

    test_dataset = VRVideo(
        root, 224, 384,
        frame_interval=1, train=False, cache_gt=True,
        transform=test_transform, pickle_file_path=metadata,
        bias_file_path=bias, dataset_name=args.dataset,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = TemporalSegFormer(PRETRAINED_MODEL, NUM_CLASSES).to(device)
    #file_path = os.path.join(save_dir, model_file)
    file_path = os.path.join(MODEL_WEIGHTS_DIR, args.dataset, model_file)
    model.load_state_dict(torch.load(file_path, map_location=device))
    print(f"\n\t Network parameters loaded from {file_path}", flush=True)

    vinfo_data = load_data(metadata)
    start = time.time()
    losses_test = []
    cc_losses_test = []
    kl_losses_test = []
    s_mse_losses_test = []
    bce_losses_test = []
    cc_test = []
    nss_test = []
    kl_test = []
    auc_judd_test = []
    sim_test = []

    for colored_image, x, last_sal, gth_sal, vid, fid, bias in test_dataloader:
        colored_image = torch.stack(colored_image, dim=1).to(device)
        last_sal = last_sal.to(device)
        gth_sal = gth_sal.to(device)

        loss, kl_loss, cc_loss, s_mse_loss, bce_loss = test_epoch(
            last_sal, gth_sal, colored_image, model, BATCH_SIZE,
            device, vid, fid, bias,
        )
        losses_test.append(loss.detach().cpu())
        cc_losses_test.append(cc_loss.detach().cpu())
        kl_losses_test.append(kl_loss.detach().cpu())
        s_mse_losses_test.append(s_mse_loss.detach().cpu())
        bce_losses_test.append(bce_loss.detach().cpu())

        predictions = model(colored_image, bias, fid).detach().cpu()
        original = gth_sal.cpu()
        g_cc, g_nss, g_kl, g_auc_judd, g_sim = evaluation_metrics(
            vinfo_data, predictions, original, vid, fid, args.dataset
        )
        cc_test.append(g_cc)
        nss_test.append(g_nss)
        kl_test.append(g_kl)
        auc_judd_test.append(g_auc_judd)
        sim_test.append(g_sim)

    concatenated_cc_test = np.concatenate(cc_test)
    concatenated_nss_test = np.concatenate(nss_test)
    concatenated_kl_test = np.concatenate(kl_test)
    concatenated_auc_judd_test = np.concatenate(auc_judd_test)
    concatenated_sim_test = np.concatenate(sim_test)

    print(f"\n Time for Testing is {time.time() - start} sec", flush=True)
    #print(f"\n\t Testing loss (single batch): {np.mean(losses_test):f}", flush=True)
    #print(f"\n\t Testing loss for cc_losses_test (single batch): {np.mean(cc_losses_test):f}", flush=True)
    #print(f"\n\t Testing loss for kl_losses_test (single batch): {np.mean(kl_losses_test):f}", flush=True)
    #print(f"\n\t Testing loss for s_mse_losses_test (single batch): {np.mean(s_mse_losses_test):f}", flush=True)
    #print(f"\n\t Testing loss for bce_losses_test (single batch): {np.mean(bce_losses_test):f}", flush=True)

    print("\n\t Average CC on Testing dataset:", f"{np.nanmean(concatenated_cc_test):.4f}", flush=True)
    print("\n\t Average NSS on Testing dataset:", f"{np.nanmean(concatenated_nss_test):.4f}", flush=True)
    print("\n\t Average KL on Testing dataset:", f"{np.nanmean(concatenated_kl_test):.4f}", flush=True)
    print("\n\t Average AUC_Judd on Testing dataset:", f"{np.nanmean(concatenated_auc_judd_test):.4f}", flush=True)
    print("\n\t Average SIM on Testing dataset:", f"{np.nanmean(concatenated_sim_test):.4f}", flush=True)

    data = {
        "losses_test": losses_test,
        "cc_losses_test": cc_losses_test,
        "kl_losses_test": kl_losses_test,
        "s_mse_losses_test": s_mse_losses_test,
        "bce_losses_test": bce_losses_test,
        "cc_test": cc_test,
        "nss_test": nss_test,
        "kl_test": kl_test,
        "auc_judd_test": auc_judd_test,
        "sim_test": sim_test,
    }
    output_path = os.path.join(save_dir, "Evaluation_metrics.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n\t Evaluation metrics saved to {output_path} \n", flush=True)


if __name__ == "__main__":
    main()
