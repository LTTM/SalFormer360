import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from dataset import VRVideo, get_dataset_config
from data_transformation import test_transform
from model import TemporalSegFormer


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
NUM_IMAGES = 50

PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 1

# Optional overrides. Leave as None to use dataset.py defaults.
DATA_ROOT = None
HEAD_DATA = None
IFCB_IMAGE = None
SAVE_DIR = "./results"
MODEL_WEIGHTS_DIR = "./model_weights"

# Image generation settings
FRAME_HEIGHT = 224
FRAME_WIDTH = 384
UPSCALE_SIZE = (1024, 512)
FIG_WIDTH = 6
DPI = 1200


def generate_and_save_images(
    model,
    colored_image,
    gth_sal,
    fid,
    ifcb_image,
    save_dir_img,
    vid_ind,
    fid_ind,
    dataset_name,
    x="Test",
):
    """Generate and save GT saliency, predicted saliency, and original images."""

    model.eval()

    os.makedirs(save_dir_img, exist_ok=True)

    with torch.no_grad():
        predictions = model(colored_image, ifcb_image, fid).cpu()

    aspect_ratio = FRAME_WIDTH / FRAME_HEIGHT
    fig_height = FIG_WIDTH / aspect_ratio

    for i in range(colored_image.shape[0]):
        video_id = vid_ind[i]
        frame_id = fid_ind[i]

        # ----------------------------------------------------
        # Ground-truth saliency
        # ----------------------------------------------------
        gth_img = np.rot90(
            np.array(gth_sal[i].cpu()).squeeze().transpose(), k=3
        )
        gth_img = np.fliplr(gth_img)
        gth_img = np.clip(gth_img, 0, 1)

        plt.figure(figsize=(FIG_WIDTH, fig_height))
        plt.imshow(gth_img, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)

        file_name = (
            f"{x}_vid{video_id}_fid{frame_id}_ground_truth_saliency.png"
        )
        file_path = os.path.join(save_dir_img, file_name)
        plt.savefig(
            file_path,
            format="png",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # ----------------------------------------------------
        # Generated saliency
        # ----------------------------------------------------
        pred_img = np.rot90(
            np.array(predictions[i].cpu()).squeeze().transpose(), k=3
        )
        pred_img = np.fliplr(pred_img)
        pred_img = np.clip(pred_img, 0, 1)

        plt.figure(figsize=(FIG_WIDTH, fig_height))
        plt.imshow(pred_img, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)

        file_name = (
            f"{x}_vid{video_id}_fid{frame_id}_generated_saliency.png"
        )
        file_path = os.path.join(save_dir_img, file_name)
        plt.savefig(
            file_path,
            format="png",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------
        # colored_image has shape [B, 2, 3, H, W].
        # Use the current frame (index 1), as in the original code.
        orig_img = np.rot90(
            np.array(colored_image[i][1].cpu()).squeeze().transpose(), k=3
        )
        orig_img = np.fliplr(orig_img)
        orig_img = np.clip(orig_img, 0, 1)
        orig_img = (orig_img * 255).astype(np.uint8)

        orig_img = Image.fromarray(orig_img).resize(
            UPSCALE_SIZE,
            Image.Resampling.LANCZOS,
        )

        plt.figure(figsize=(FIG_WIDTH, fig_height))
        plt.imshow(orig_img)
        plt.axis("off")
        plt.tight_layout(pad=0)

        file_name = (
            f"{x}_vid{video_id}_fid{frame_id}_original_image.png"
        )
        file_path = os.path.join(save_dir_img, file_name)
        plt.savefig(
            file_path,
            format="png",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    print(f"Images saved to: {save_dir_img}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate GT, predicted saliency, and original images."
    )

    parser.add_argument(
        "--dataset",
        choices=["360", "eye", "pvs"],
        default=DATASET,
    )
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--metadata", default=HEAD_DATA)
    parser.add_argument("--bias", default=IFCB_IMAGE)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------
    config = get_dataset_config(args.dataset)

    root = args.root or config["root"]
    metadata = args.metadata or config["pickle_file"]
    bias = args.bias or config["bias_file"]

    # Automatically select the final model for the dataset.
    model_file = args.model or MODEL_FILES[args.dataset]

    # Keep generated images separated by dataset.
    save_dir_img = os.path.join(
        args.save_dir,
        args.dataset,
        "images",
    )
    os.makedirs(save_dir_img, exist_ok=True)

    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Root: {root}", flush=True)
    print(f"Metadata: {metadata}", flush=True)
    print(f"IFCB image: {bias}", flush=True)
    print(f"Model: {model_file}", flush=True)
    print(f"Image output directory: {save_dir_img}", flush=True)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}", flush=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    test_dataset = VRVideo(
        root,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        frame_interval=1,
        train=False,
        cache_gt=True,
        transform=test_transform,
        pickle_file_path=metadata,
        bias_file_path=bias,
        dataset_name=args.dataset,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("The selected test dataset contains no samples.")

    num_images = min(args.num_images, len(test_dataset))

    rng = np.random.default_rng(seed)
    image_inds = rng.choice(
        len(test_dataset),
        num_images,
        replace=False,
    )

    print(
        f"Generating images for {num_images} samples "
        f"out of {len(test_dataset)} test samples.",
        flush=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = TemporalSegFormer(PRETRAINED_MODEL, NUM_CLASSES)
    #model_path = os.path.join(args.save_dir, model_file)
    model_path = os.path.join(MODEL_WEIGHTS_DIR, args.dataset, model_file)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}\n"
            f"Use --model or place the expected checkpoint in --save-dir."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Network parameters loaded from: {model_path}", flush=True)

    # --------------------------------------------------------
    # Collect samples
    # --------------------------------------------------------
    samples = [test_dataset[i] for i in image_inds]

    colored_image = torch.stack(
        [torch.stack(sample[0], dim=0) for sample in samples],
        dim=0,
    ).to(device)

    gth_sal = torch.stack(
        [sample[3] for sample in samples],
        dim=0,
    ).to(device)

    fid = torch.tensor(
        [sample[5] for sample in samples],
        dtype=torch.long,
        device=device,
    )

    ifcb_image = torch.stack(
        [sample[6] for sample in samples],
        dim=0,
    ).to(device)

    vid_ind = [sample[4] for sample in samples]
    fid_ind = [sample[5] for sample in samples]

    # --------------------------------------------------------
    # Generate and save images
    # --------------------------------------------------------
    generate_and_save_images(
        model=model,
        colored_image=colored_image,
        gth_sal=gth_sal,
        fid=fid,
        ifcb_image=ifcb_image,
        save_dir_img=save_dir_img,
        vid_ind=vid_ind,
        fid_ind=fid_ind,
        dataset_name=args.dataset,
        x="Test",
    )


if __name__ == "__main__":
    main()
