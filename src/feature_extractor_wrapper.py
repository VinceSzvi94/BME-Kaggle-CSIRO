import torch
import torch.nn as nn
from torchvision import models

MODELS = [
    # MobileNet variants
    'MobileNetV2',
    'MobileNetV3-Small',
    'MobileNetV3-Large',
    # EfficientNet variants (available in torchvision)
    'EfficientNet-B0',
    'EfficientNet-B1',
    'EfficientNet-B2',
    'EfficientNet-B3',
    'EfficientNet-B4',
    # ShuffleNet variants
    'ShuffleNet-V2-x0.5',
    'ShuffleNet-V2-x1.0',
    'ShuffleNet-V2-x1.5',
    'ShuffleNet-V2-x2.0',
    # RegNet variants
    'RegNet-Y-400MF',
    'RegNet-Y-800MF',
    'RegNet-Y-1.6GF',
    'RegNet-Y-3.2GF',
]

def modify_classifier(model, trainable=False):
    """
    Removes the last classifier block of a torchvision model.
    """
    # Set all parameters to not trainable if specified
    if not trainable:
        for param in model.parameters():
            param.requires_grad = False

    # 1. Handle Models with 'fc' (ShuffleNet, RegNet, ResNet)
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        # Replace
        model.fc = nn.Identity()
        
    # 2. Handle Models with 'classifier' (MobileNet, EfficientNet, VGG)
    elif hasattr(model, 'classifier'):
        # Case A: Classifier is just a Linear layer (rare in these specific models but possible)
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = 0
            
        # Case B: Classifier is a Sequential block (MobileNetV2, V3, EfficientNet)
        elif isinstance(model.classifier, nn.Sequential):
            # We need to find the input features.
            # Usually, we can look at the first Linear layer inside the block.
            # OR the last Linear layer's input features (if simple Dropout->Linear).
            
            # Logic for MobileNetV3 (starts with Linear)
            if isinstance(model.classifier[0], nn.Linear):
                in_features = model.classifier[0].in_features
            
            # Logic for MobileNetV2 / EfficientNet (starts with Dropout, then Linear)
            # In these, the Linear layer is usually the last item [-1]
            else:
                for layer in model.classifier:
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        break
            
            # Replace the ENTIRE classifier block
            model.classifier = nn.Identity()
    
    return model, in_features

class FeatureExtractorWrapper(nn.Module):
    def __init__(self, model_name: str, target_dim=128, dropout_rate=0.2, trainable=False):
        super(FeatureExtractorWrapper, self).__init__()
        
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from: {MODELS}")
        
        # Load the pre-trained model
        model = None
        if model_name == 'MobileNetV2':
            model = models.mobilenet_v2(pretrained=True)
        elif model_name == 'MobileNetV3-Small':
            model = models.mobilenet_v3_small(pretrained=True)
        elif model_name == 'MobileNetV3-Large':
            model = models.mobilenet_v3_large(pretrained=True)
        elif model_name == 'EfficientNet-B0':
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == 'EfficientNet-B1':
            model = models.efficientnet_b1(pretrained=True)
        elif model_name == 'EfficientNet-B2':
            model = models.efficientnet_b2(pretrained=True)
        elif model_name == 'EfficientNet-B3':
            model = models.efficientnet_b3(pretrained=True)
        elif model_name == 'EfficientNet-B4':
            model = models.efficientnet_b4(pretrained=True)
        elif model_name == 'ShuffleNet-V2-x0.5':
            model = models.shufflenet_v2_x0_5(pretrained=True)
        elif model_name == 'ShuffleNet-V2-x1.0':
            model = models.shufflenet_v2_x1_0(pretrained=True)
        elif model_name == 'ShuffleNet-V2-x1.5':
            model = models.shufflenet_v2_x1_5(pretrained=True)
        elif model_name == 'ShuffleNet-V2-x2.0':
            model = models.shufflenet_v2_x2_0(pretrained=True)
        elif model_name == 'RegNet-Y-400MF':
            model = models.regnet_y_400mf(pretrained=True)
        elif model_name == 'RegNet-Y-800MF':
            model = models.regnet_y_800mf(pretrained=True)
        elif model_name == 'RegNet-Y-1.6GF':
            model = models.regnet_y_1_6gf(pretrained=True)
        elif model_name == 'RegNet-Y-3.2GF':
            model = models.regnet_y_3_2gf(pretrained=True)

        self.model_name = model_name + f"_and_lin{target_dim}"
        base_model, in_features = modify_classifier(model, trainable)
        self.base_model = base_model
        self.new_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, target_dim)
        )
        self.trainable = trainable
    
    def get_architecture_name(self):
        return self.model_name
    
    def forward(self, x):
        x_base = self.base_model(x)
        return self.new_layer(x_base)