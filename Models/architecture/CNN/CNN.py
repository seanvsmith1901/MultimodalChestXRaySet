class FinetunedDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load DenseNet-121 with pretrained ImageNet weights
        densenet = models.densenet121(weights="IMAGENET1K_V1")
        dropout = 0.5
        
        # Replace first conv with 1-channel version
        old_conv = densenet.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,  # Grayscale
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Average RGB weights → Grayscale weights
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
        
        densenet.features.conv0 = new_conv
        self.features = densenet.features
        num_features = 1024
    
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 1)
        )
    def forward(self, x):
        x = self.features(x)      # x shape: (batch, 3, 224, 224) → (batch, 1024, 7, 7)
        x = self.classifier(x)    # → (batch, 1)
        return x