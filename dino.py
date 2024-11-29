import torch
import torch.nn as nn
import torch.nn.functional as F

class DETREncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, ff_dim=2048):
        super(DETREncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x (batch_size, H*W, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DETRTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_queries=100, ff_dim=2048):
        super(DETRTransformer, self).__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.positional_encoding = nn.Parameter(torch.randn(1, num_queries, embed_dim))

        self.class_logits = nn.Linear(embed_dim, num_classes)  #! need Number of classes
        self.bbox_pred = nn.Linear(embed_dim, 4)

    def forward(self, x):
        # x (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()

        # Add positional encoding
        x = x + self.positional_encoding

        # Multihead Attention (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)

        # Feed-forward network
        x = self.ffn(x)

        class_logits = self.class_logits(x)  # (batch_size, num_queries, num_classes)
        bbox_pred = self.bbox_pred(x)  # (batch_size, num_queries, 4)

        return class_logits, bbox_pred

class DETR(nn.Module):
    def __init__(self, backbone, num_classes, embed_dim=512, num_queries=100, num_heads=8, ff_dim=2048, num_encoder_layers=6):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.encoder = DETREncoder(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_encoder_layers, ff_dim=ff_dim)
        self.transformer = DETRTransformer(num_classes, embed_dim=embed_dim, num_heads=num_heads, num_queries=num_queries, ff_dim=ff_dim)
    
    def forward(self, x):
        features = self.backbone(x)
        batch_size, channels, height, width = features.size()
        features = features.flatten(2).permute(0, 2, 1)  # (batch_size, H*W, channels)
        encoder_output = self.encoder(features)
        class_logits, bbox_pred = self.transformer(encoder_output)
        return class_logits, bbox_pred