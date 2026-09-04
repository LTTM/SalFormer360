import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig

# Define Modified SegFormer Decoder
class ModifiedSegFormerDecoder(nn.Module):
    def __init__(self, encoder_output_shapes, num_classes):
        super(ModifiedSegFormerDecoder, self).__init__()

        # First convolution to process the encoder's highest-resolution output
        self.conv1 = nn.Conv2d(
            encoder_output_shapes[-1], 128, kernel_size=(3, 3), stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # Second convolution and upsampling
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )

        # Third convolution and upsampling
        self.conv3 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
       
        # Apply final convolution to produce the saliency map
        self.final_conv = nn.Conv2d(
            16, num_classes, kernel_size=(3, 3), stride=1, padding=1
        )

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, encoder_outputs):
        """
        Decoder forward pass to reconstruct the saliency map.

        Args:
            encoder_outputs: List of feature maps from the encoder.

        Returns:
            Saliency map at input resolution.
        """
        # Start with the highest-resolution feature map from the encoder
        x = encoder_outputs[-1]

        # Apply convolutions, batch normalization, activations, and progressive upsampling
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.upsample1(x)
        x = self.dropout(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.upsample2(x)
        x = self.dropout(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.upsample3(x)
        x = self.dropout(x)

        # Apply the final convolution to produce the saliency map
        saliency_map = self.final_conv(x)
        saliency_map = self.sigmoid(saliency_map)

        return saliency_map


class TemporalSegFormer(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(TemporalSegFormer, self).__init__()

        # Load pretrained SegFormer model
        config = SegformerConfig.from_pretrained(
            pretrained_model_name, output_hidden_states=True
        )

        # Modify config to accept 6-channel input
        config.num_channels = 6  # Update input channels to 6
        self.encoder = SegformerModel(config)

        # Extract encoder hidden sizes for the decoder
        encoder_output_shapes = config.hidden_sizes

        # Initialize the decoder
        self.decoder = ModifiedSegFormerDecoder(encoder_output_shapes, num_classes)

        # IFCB learnable parameter (α)
        self.alpha = nn.Parameter(
            torch.tensor(600.0, requires_grad=True)
        )  # Trainable decay factor

        self.beta = nn.Parameter(
            torch.tensor(0.15, requires_grad=True)
        )  

        # Load pretrained weights and adapt input layer
        self._load_pretrained_weights(pretrained_model_name)

    def _load_pretrained_weights(self, pretrained_model_name):
        # Load pretrained state dict
        pretrained_state_dict = SegformerModel.from_pretrained(
            pretrained_model_name
        ).state_dict()

        # Adapt the first conv layer in patch_embeddings to accept 6 channels
        patch_proj_weight = pretrained_state_dict[
            "encoder.patch_embeddings.0.proj.weight"
        ]  # Shape: [32, 3, 7, 7]
        new_weight = torch.zeros(
            32, 6, 7, 7
        )  # Initialize new weight tensor with 6 input channels
        new_weight[:, :3, :, :] = (
            patch_proj_weight  # Copy pretrained weights for the first 3 channels
        )
        new_weight[:, 3:, :, :] = patch_proj_weight[
            :, :3, :, :
        ]  # Duplicate weights for the additional 3 channels
        pretrained_state_dict["encoder.patch_embeddings.0.proj.weight"] = new_weight

        # Load the adapted state dict into the model
        self.encoder.load_state_dict(pretrained_state_dict, strict=False)

    def gaussian_decay_weight(self, fid, batch_size, C=600, device="cuda"):
        """
        Compute time-dependent decay weight w_t using Gaussian function.
        Args:
            fid (Tensor): Tensor of frame indices.
            batch_size (int): Batch size.
            C (int): Constant for normalization.
            device (str): Device to run on (CPU/GPU).
        Returns:
            Tensor: Batch-wise decay weight (B,1,1,1).
        """
        fid = fid.float().to(device)  # Convert to float tensor (batch_size,)
        wt = torch.exp(-self.alpha * (fid / C) ** 2)  # Compute decay per frame
        return wt.view(batch_size, 1, 1, 1)  # Reshape to (B,1,1,1)

    def forward(self, x, ifcb_map, fid):
        """
        Args:
            x: Tensor of shape (B, T, C, H, W), where T=2 and C=3.
            ifcb_map: Precomputed IFCB map of shape (1,1,H,W).
            fid: Frame indices.
        Returns:
            Saliency map at input resolution.
        """
        # Input shape: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        if t != 2 or c != 3:
            raise ValueError(
                f"Expected input with shape (B, 2, 3, H, W), but got {x.shape}"
            )

        # Reshape to (B, 6, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # Reorder dimensions to (B, C, T, H, W)
        x = x.reshape(
            b, c * t, h, w
        )  # Combine temporal and channel dimensions: (B, 6, H, W)

        # Pass through the encoder
        outputs = self.encoder(x)
        encoder_outputs = outputs.hidden_states  # Get encoder features

        # Decode to produce final saliency map
        decoder_output = self.decoder(encoder_outputs)  # Shape: (B,1,H,W)

        # Compute adaptive weight wt based on frame ID
        wt = self.gaussian_decay_weight(fid, batch_size=b, device=x.device)  # (B,1,1,1)

        # Expand IFCB map to match batch size
        ifcb_batch = ifcb_map.expand(b, -1, -1, -1).to(x.device).detach()  # (B,1,H,W)
       
        final_output = wt * ifcb_batch + (1 - wt) * decoder_output  # (B,1,H,W)

        beta = torch.clamp(self.beta, 0.0, 1.0)

        final_output = (final_output * (1 - beta)) + (ifcb_batch * beta)

        return final_output

