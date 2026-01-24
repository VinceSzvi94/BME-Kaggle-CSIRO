import torch
import torch.nn as nn

from src.feature_extractor_wrapper import FeatureExtractorWrapper

DINO_MODELS = ['dinov2_vits14', 'dinov2_vitb14']

class HybridModel(nn.Module):
    def __init__(self, dino_model_name: str, fe_model: FeatureExtractorWrapper, tilesize=224, overlap=0, normalize_features=True, reg_layers=[512, 256], reg_activation=nn.ReLU(), num_outputs=5):
        super(HybridModel, self).__init__()
        
        # Enforce overlap=0 for summation tasks to avoid double-counting
        if overlap > 0:
            print("Warning: Overlap > 0 in a summation task will result in double-counting biomass!")
        
        if tilesize % 14 != 0:
            print("Warning: Tilesize should be a multiple of 14 for DINOv2.")
        
        if  dino_model_name not in DINO_MODELS:
            raise ValueError(f"Unsupported DINO model: {dino_model_name}. Supported models: {DINO_MODELS}")

        # vit model
        self.dino_model_name = dino_model_name
        dino_model = torch.hub.load('facebookresearch/dinov2', dino_model_name)
        for param in dino_model.parameters():
            param.requires_grad = False
        self.dino_model = dino_model

        # cnn model
        self.fe_model = fe_model

        # final regression layers
        if hasattr(fe_model.model, 'fc'):
            fe_output_dim = fe_model.model.fc[-1].out_features
        elif hasattr(fe_model.model, 'classifier'):
            fe_output_dim = fe_model.model.classifier[-1].out_features
        else:
            raise AttributeError("Feature extractor model has neither 'fc' nor 'classifier' attribute")
        final_input_dim = self.dino_model.embed_dim + fe_output_dim
        reg_layers_2 = [final_input_dim] + reg_layers
        layer_components = []
        for i in range(len(reg_layers)):
            layer_components.append(nn.Linear(reg_layers_2[i], reg_layers_2[i+1]))
            layer_components.append(reg_activation)
            layer_components.append(nn.Dropout(0.5 if i == 0 else 0.3))

        self.reg_layers = nn.Sequential(
            nn.Flatten(),
            *layer_components,
            nn.Linear(reg_layers_2[-1], num_outputs)  # No activation for regression
        )
        self.reg_layer_sizes = reg_layers
        self.normalize_features = normalize_features
        self.tile_size = tilesize
        self.unfold = nn.Unfold(kernel_size=tilesize, stride=tilesize-overlap)
    
    # model name for logging
    def get_architecture_name(self):
        return f"Hybrid_{self.dino_model_name}_{self.fe_model.get_architecture_name()}_{self.tile_size}x{self.tile_size}_reg{'-'.join(map(str, self.reg_layer_sizes))}"

    def forward(self, img):
        # img shape: (Batch, 3, H, W)
        # print(f"Debug: Input image shape: {img.shape}")
        
        # 1. Unfold into patches
        # Output shape: (B, C*tilesize*tilesize, N_tiles)
        x = self.unfold(img)  
        # print(f"Debug: shape after unfold: {x.shape}")
        
        # 2. Reshape to be a batch of standard images
        x = x.transpose(1, 2) # (B, N_tiles, C*tilesize*tilesize)
        b, n_patches, features = x.shape
        
        # Collapse Batch and N_tiles together for the CNN
        x = x.reshape(b * n_patches, 3, self.tile_size, self.tile_size)
        # print(f"Debug: Input to model shape: {x.shape}")
        # for i in range(x.shape[0]):
        #     # save image for debugging
        #     from torchvision.utils import save_image
        #     save_image(x[i], f"debug/debug_input_tile_{i}.png")
        
        # 3. Pass through the model
        # batched_out shape: (Batch * N_tiles, output_dim)
        batched_out_vit = self.dino_model(x)
        batched_out_fe = self.fe_model(x)
        
        # 4. Separate Batch and Tiles again
        # Shape: (Batch, N_tiles, output_dim)
        out_unflat_vit = batched_out_vit.reshape(b, n_patches, -1)
        out_unflat_fe = batched_out_fe.reshape(b, n_patches, -1)
        # print(f"Debug: Output from model shape (before summation): {out_unflat.shape}")
        
        # Sum over the tiles (dim 1)
        # We sum across the 'n_patches' dimension to get total grams per image
        avg_out_vit = out_unflat_vit.mean(dim=1) # Shape: (Batch, output_dim)
        avg_out_fe = out_unflat_fe.mean(dim=1) # Shape: (Batch, output_dim)

        if self.normalize_features:
            # L2 normalize both feature vectors
            avg_out_vit = nn.functional.normalize(avg_out_vit, p=2, dim=1)
            avg_out_fe = nn.functional.normalize(avg_out_fe, p=2, dim=1)

        # 5. Concatenate vit and fe features
        concat_out = torch.cat((avg_out_vit, avg_out_fe), dim=1)  # Shape: (Batch, vit_dim + fe_dim)

        # 6. Final regression layers
        out = self.reg_layers(concat_out)  # Shape: (Batch, num_outputs)
        
        return out