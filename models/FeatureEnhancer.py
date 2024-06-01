import torch
import torch.nn as nn
from torch.nn import functional as F
from DeformableAtt import MultiScaleDeformableAttention as DeformableAttention
from Backbones import MultiModalFeatureExtractor


class FeatureEnhancerLayer(nn.Module):
    def __init__(self, d_model,  num_heads=8, num_levels=4, num_points=4, img2col_step= 64,):
        super(FeatureEnhancerLayer, self).__init__()
        self.image_self_attn = DeformableAttention(d_model, num_heads=num_heads, num_levels=num_levels, num_points=num_points, img2col_step=64, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(d_model, num_heads=num_heads)
        self.img_to_text_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads)
        self.text_to_img_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, img_features, text_features, spatial_shapes, reference_points):
        img_features = self.image_self_attn(img_features, img_features, img_features, spatial_shapes= spatial_shapes, reference_points=reference_points)[0]
        text_features = self.text_self_attn(text_features, text_features, text_features)[0]
        img_to_text_features = self.img_to_text_cross_attn(text_features, img_features, img_features)[0]
        text_to_img_features = self.text_to_img_cross_attn(img_features, text_features, text_features)[0]
        img_features = self.ffn(img_features + text_to_img_features)
        text_features = self.ffn(text_features + img_to_text_features)
        return img_features, text_features


# Example usage
device='cpu'
model = FeatureEnhancerLayer(1024)
images = torch.randn(2, 3, 224, 224) # Example batch of images
texts = ["a photo of a cat", "a picture of a dog"]  # Example batch of text descriptions
bs = 2
num_query = 4165
num_levels = 4
reference_points = torch.rand(bs, num_query, num_levels, 2)
backbone = MultiModalFeatureExtractor()
text_features, img_features = backbone(texts, images)

#Img_features
spatial_shapes = []
feat_flatten = []
num_backbone_outs = 4
input_proj_list = []
hidden_dim = 1024
for _ in range(len(img_features)):
    in_channels = img_features[_].shape[-1]
    input_proj_list.append(
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        )
    )
input_proj = nn.ModuleList(input_proj_list)
for l, feat in enumerate(img_features):
    B, H, W, C = feat.shape
    feat_flatten.append(input_proj[l](feat.permute(0,3,1,2)).permute(0, 2, 3, 1).flatten(1,2))
    spatial_shapes.append([H, W])
spatial_shapes = torch.tensor(spatial_shapes)
feat_flatten = torch.cat(feat_flatten, 1)

#Text_features
bert_size = 768
feat_map = nn.Linear(bert_size, hidden_dim, bias=True)
text_features = feat_map(text_features)
#Feature Enhancer
img_features, text_features = model(feat_flatten, text_features, spatial_shapes, reference_points)
print(img_features.shape, text_features.shape)