import os
import numbers
import pickle
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from random import Random


# ============================================================
# Dataset-specific default settings
# ============================================================
DATASET_CONFIGS = {
    "360": {
        "root": "./datasets/360/360_Saliency_dataset_2018ECCV",
        "pickle_file": "./datasets/360/360_Saliency_dataset_2018ECCV/vinfo.pkl",
        "bias_file": "./biases/360_bias.png",
        "split": "random",
    },
    "eye": {
        "root": "./datasets/eye/VR-EyeTracking",
        "pickle_file": "./datasets/eye/3VR-EyeTracking/head_data.pkl",
        "bias_file": "./biases/eye_bias.png",
        "split": "eye",
    },
    "pvs": {
        "root": "./datasets/pvs/PVS-HM",
        "pickle_file": "./datasets/pvs/gaze_data_from_FULLdata.pkl",
        "bias_file": "./biases/pvs_bias.png",
        "split": "pvs",
    },
}


# The fixed training split used by the VR-EyeTracking code.
EYE_TRAIN_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
    163, 164, 165, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
    189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
]

EYE_TRAIN_IDS = {f"{i:03d}" for i in EYE_TRAIN_IDS}


# The fixed training split used by the PVS-HM code.
PVS_TRAIN_IDS = {
    "A380", "AcerEngine", "AcerPredator", "AirShow", "BFG", "Bicycle",
    "Camping", "CandyCarnival", "Castle", "Catwalks", "CMLauncher", "CS",
    "DanceInTurn", "DrivingInAlps", "Egypt", "F5Fighter", "Flight",
    "GalaxyOnFire", "Graffiti", "GTA", "HondaF1", "IRobot", "KasabianLive",
    "Lion", "LoopUniverse", "Manhattan", "MC", "MercedesBenz", "Motorbike",
    "Murder", "Orion", "Parachuting", "Parasailing", "Pearl", "Predator",
    "ProjectSoul", "Rally", "RingMan", "Roma", "Shark", "Skiing", "Snowfield",
    "SnowRopeway", "Square", "StarWars", "StarWars2", "Stratosphere",
    "StreetFighter", "Supercar", "SuperMario64", "Surfing", "SurfingArctic",
    "TalkingInCar", "Terminator", "TheInvisible", "Village", "VRBasketball",
    "Waterskiing", "WesternSichuan", "Yacht",
}


def get_dataset_config(dataset_name):
    """Return the default paths and split configuration for a dataset."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_CONFIGS)}"
        )
    return DATASET_CONFIGS[dataset_name].copy()


class VRVideo(data.Dataset):
    """Dataset loader shared by the three 360-degree saliency datasets."""

    def __init__(
        self,
        root,
        frame_h,
        frame_w,
        frame_interval=5,
        transform=None,
        train=True,
        cache_gt=True,
        rnd_seed=367643,
        pickle_file_path=None,
        bias_file_path=None,
        dataset_name="360",
    ):
        self.frame_interval = frame_interval
        self.transform = transform
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.cache_gt = cache_gt
        self.train = train
        self.dataset_name = dataset_name.lower()

        config = get_dataset_config(self.dataset_name)
        rnd = Random(rnd_seed)

        if pickle_file_path is None:
            pickle_file_path = config["pickle_file"]
        if bias_file_path is None:
            bias_file_path = config["bias_file"]

        self.bias = Image.open(bias_file_path).convert("L")

        with open(pickle_file_path, "rb") as f:
            vinfo = pickle.load(f)

        existing_folders = os.listdir(root)
        filtered_vinfo = {
            key: value for key, value in vinfo.items() if key in existing_folders
        }
        self.vinfo = filtered_vinfo

        vset = sorted(
            [vid for vid in os.listdir(root) if os.path.isdir(os.path.join(root, vid))]
        )

        print(f"{len(vset)} videos found.", flush=True)

        if set(self.vinfo.keys()) != set(vset):
            print("Warning: dataset folders and metadata keys do not match.", flush=True)
            print("In folders but not metadata:", set(vset) - set(self.vinfo.keys()), flush=True)
            print("In metadata but not folders:", set(self.vinfo.keys()) - set(vset), flush=True)

        # ------------------------------------------------------------
        # Dataset-specific train/test split
        # ------------------------------------------------------------
        if self.dataset_name == "eye":
            train_videos = sorted(EYE_TRAIN_IDS & set(vset))
            val_videos = sorted(set(vset) - EYE_TRAIN_IDS)

        elif self.dataset_name == "pvs":
            train_videos = sorted(PVS_TRAIN_IDS & set(vset))
            val_videos = sorted(set(vset) - PVS_TRAIN_IDS)

        elif self.dataset_name == "360":
            # randomly select video_train videos for training.
            
            video_train = 80

            if video_train > len(vset):
                raise ValueError(
                    f"video_train={video_train} is larger than the {len(vset)} available videos."
                )
            train_videos = sorted(rnd.sample(vset, k=video_train))
            val_videos = sorted(set(vset) - set(train_videos))

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        print(
            f"{len(train_videos)}:{len(val_videos)} videos chosen for training:testing.",
            flush=True,
        )
        print(train_videos, flush=True)
        print(val_videos, flush=True)

        vset = train_videos if train else val_videos

        self.data = []
        self.target = []
        self.i2v = {}
        self.v2i = {}

        for vid in vset:
            obj_path = os.path.join(root, vid)
            frame_list = [
                frame for frame in os.listdir(obj_path) if frame.endswith(".jpg")
            ]
            frame_list.sort()

            for i, frame in enumerate(frame_list):
                if i % self.frame_interval == 0:
                    fid = frame[:-4]
                    if fid not in self.vinfo.get(vid, {}):
                        continue
                    self.i2v[len(self.data)] = (vid, fid)
                    self.v2i[(vid, fid)] = len(self.data)
                    self.data.append(os.path.join(obj_path, frame))
                    self.target.append(self.vinfo[vid][fid])

        if not self.target:
            raise RuntimeError(
                f"No frames/targets found for dataset='{self.dataset_name}', train={train}."
            )

    def __getitem__(self, item):
        vid, fid = self.i2v[item]
        fid = int(fid)

        # Define the sequence of frame indices [t-5, t].
        frame_indices = [fid + offset for offset in [-5, 0]]

        valid_frame_indices = [int(f) for v, f in self.i2v.values() if v == vid]
        if not valid_frame_indices:
            raise ValueError(f"No valid frames found for video {vid}")

        first_frame_index = min(valid_frame_indices)
        last_frame_index = max(valid_frame_indices)

        frames = []
        for frame_idx in frame_indices:
            clamped_frame_idx = max(
                first_frame_index, min(frame_idx, last_frame_index)
            )
            frame_key = (vid, f"{clamped_frame_idx:04d}")

            if frame_key not in self.v2i:
                print(
                    f"Warning: Frame key {frame_key} not found in v2i. "
                    "Filling with a blank frame."
                )
                blank_frame = torch.zeros(
                    (3, self.frame_h, self.frame_w), dtype=torch.float32
                )
                frames.append(blank_frame)
            else:
                img_path = self.data[self.v2i[frame_key]]
                try:
                    with open(img_path, "rb") as img_file:
                        img = Image.open(img_file).convert("RGB")
                    frames.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    blank_frame = torch.zeros(
                        (3, self.frame_h, self.frame_w), dtype=torch.float32
                    )
                    frames.append(blank_frame)

        while len(frames) < 2:
            print("Warning: Not enough frames collected. Filling with a blank frame.")
            blank_frame = torch.zeros(
                (3, self.frame_h, self.frame_w), dtype=torch.float32
            )
            frames.append(blank_frame)

        last_frame_key = (
            vid,
            f"{max(fid - self.frame_interval, first_frame_index):04d}",
        )
        last = self._get_salency_map(
            self.v2i.get(last_frame_key, first_frame_index)
        )

        target = self._get_salency_map(item)
        bias = self.bias.copy()

        if self.transform:
            frames[0], frames[1], target, bias = self.transform(
                frames[0], frames[1], target, bias, self.dataset_name
            )

        if self.train:
            return frames, last, target, vid, fid, bias
        else:
            return frames, self.data[item], last, target, vid, fid, bias

    def __len__(self):
        return len(self.data)

    def _get_salency_map(self, item, use_cuda=False):
        cfile = self.data[item][:-4] + '_gt.npy'
        if item >= 0:
            if self.cache_gt and os.path.isfile(cfile):
                target_map = torch.from_numpy(np.load(cfile)).float()
                #assert target_map.size() == (1, self.frame_h, self.frame_w)
                return torch.from_numpy(np.load(cfile)).float()

        else:
          raise ValueError(f"Invalid item index: {item}")

    
    