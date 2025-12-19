import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class ResNet_RB(nn.Module):
    def __init__(self, num_res_blocks=8, input_size=(64, 64), num_outputs=5):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        # Calculate flattened size
        flattened_size = 64 * input_size[0] * input_size[1]
        
        # Linear layers for regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)  # No activation for regression
        )
    
    # this function is to identify the architecture in logging
    def get_architecture_name(self):
        return f"ResNet_RB_{len(self.res_blocks)}blocks"

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.mid(x) + initial
        x = self.fc(x)    # Shape: [batch, num_outputs]
        return x