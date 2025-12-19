import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, model, tilesize=64, overlap=0):
        super(ModelWrapper, self).__init__()
        
        # Enforce overlap=0 for summation tasks to avoid double-counting
        if overlap > 0:
            print("Warning: Overlap > 0 in a summation task will result in double-counting biomass!")
            
        self.model = model
        self.tile_size = tilesize
        self.unfold = nn.Unfold(kernel_size=tilesize, stride=tilesize-overlap)
    
    # model name for logging
    def get_architecture_name(self):
        if hasattr(self.model, 'get_architecture_name'):
            return self.model.get_architecture_name()
        else:
            return "Unknown Model"

    def forward(self, img):
        # img shape: (Batch, 3, H, W)
        
        # 1. Unfold into patches
        # Output shape: (B, C*tilesize*tilesize, N_tiles)
        x = self.unfold(img)  
        
        # 2. Reshape to be a batch of standard images
        x = x.transpose(1, 2) # (B, N_tiles, C*tilesize*tilesize)
        b, n_patches, features = x.shape
        
        # Collapse Batch and N_tiles together for the CNN
        x = x.view(b * n_patches, 3, self.tile_size, self.tile_size)
        
        # 3. Pass through the model
        # batched_out shape: (Batch * N_tiles, 5)
        batched_out = self.model(x)
        
        # 4. Separate Batch and Tiles again
        # Shape: (Batch, N_tiles, 5)
        out_unflat = batched_out.view(b, n_patches, -1)
        
        # Sum over the tiles (dim 1)
        # We sum across the 'n_patches' dimension to get total grams per image
        out = out_unflat.sum(dim=1) # Shape: (Batch, 5)
        
        return out