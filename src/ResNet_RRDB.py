import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    """
    The internal Dense Block used inside the RRDB. Based on ESRGAN github code.
    """
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, nf: number of filters
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    """
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Residual scaling (paper suggests scaling by 0.2)
        return out * 0.2 + x

class ResNet_RRDB(nn.Module):
    def __init__(self, in_channels=3, num_res_blocks=8, input_size=(64, 64), linear_layers=[512, 256], num_outputs=5): 
        # ESRGAN typically uses 23 blocks (vs 16 in SRGAN)
        super(ResNet_RRDB, self).__init__()

        self.initial = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # RRDB blocks
        self.res_blocks = nn.Sequential(
            *[RRDB(nf=64, gc=32) for _ in range(num_res_blocks)]
        )

        self.conv_body = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Calculate flattened size
        flattened_size = 64 * input_size[0] * input_size[1]
        
        linear_layers_2 = [flattened_size] + linear_layers
        layer_components = []
        for i in range(len(linear_layers)):
            layer_components.append(nn.Linear(linear_layers_2[i], linear_layers_2[i+1]))
            layer_components.append(nn.ReLU())
            layer_components.append(nn.Dropout(0.5 if i == 0 else 0.3))
        
        # Linear layers for regression
        self.fc = nn.Sequential(
            nn.Flatten(),
            *layer_components,
            nn.Linear(linear_layers_2[-1], num_outputs)  # No activation for regression
        )
    
    def get_architecture_name(self):
        return f"ResNet_RRDB_{len(self.res_blocks)}blocks"

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.conv_body(x) + initial # Global skip connection
        x = self.fc(x)    # Shape: [batch, num_outputs]
        return x


