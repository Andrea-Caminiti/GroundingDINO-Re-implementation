import torch
import torch.nn as nn
from torch.nn import functional as F
from DeformableAtt import MultiScaleDeformableAttention as DeformableAttention
from Backbones import MultiModalFeatureExtractor


class FeatureEnhancerLayer(nn.Module):
    '''
    
    Feature Enhancer Layer that takes in input images and texts and return image features and text features
    
    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the attentions. Default 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default 4.
    '''
    def __init__(self, d_model,  num_heads=8, num_levels=4, num_points=4):
        super(FeatureEnhancerLayer, self).__init__()
        self.d_model = d_model
        self.image_self_attn = DeformableAttention(d_model, num_heads=num_heads, num_levels=num_levels, num_points=num_points, img2col_step=64, batch_first=True)
        self.text_self_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.img_to_text_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.text_to_img_cross_attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.backbone = MultiModalFeatureExtractor()
        self.input_proj_list = []
        self.bert_size = 768
        self.feat_map = nn.Linear(self.bert_size, self.d_model, bias=True)
    def forward(self, images, texts, reference_points):
        '''
        Args:
            images: batch of images from which to extract features
            texts: batch of texts from which to extract features
            reference_points: The normalized reference points
                              with shape `(bs, num_query, num_levels, 2)`,
                              all elements is range in [0, 1], top-left (0, 0),
                              bottom-right (1, 1), including padding are.
                              or `(bs, num_query, num_levels, 4)`, add additional
                              two dimensions `(h, w)` to form reference boxes.
        Returns:
            img_features: image features tensor with shape (bs, seq_lenght, d_model)
            text_features: text features tensor with shape (bs, seq_lenght, d_model)
        '''
        text_features, img_features = self.backbone(texts, images)

        #Img_features
        spatial_shapes = []
        feat_flatten = []
        for _ in range(len(img_features)): 
            in_channels = img_features[_].shape[-1]
            self.input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            )
        self.input_proj = nn.ModuleList(self.input_proj_list)
        for l, feat in enumerate(img_features):
            B, H, W, C = feat.shape
            feat_flatten.append(self.input_proj[l](feat.permute(0,3,1,2)).permute(0, 2, 3, 1).flatten(1,2))
            spatial_shapes.append([H, W])
        spatial_shapes = torch.tensor(spatial_shapes)
        img_features = torch.cat(feat_flatten, 1)

        #Text_features
        text_features = self.feat_map(text_features)

        img_features = self.image_self_attn(img_features, img_features, img_features, spatial_shapes= spatial_shapes, reference_points=reference_points) #bs x img_seq_lenght x d_model
        text_features = self.text_self_attn(text_features, text_features, text_features)[0] #bs x text_seq_lenght x d_model
        img_to_text_features = self.img_to_text_cross_attn(text_features, img_features, img_features)[0] #bs x text_seq_lenght x d_model
        text_to_img_features = self.text_to_img_cross_attn(img_features, text_features, text_features)[0]#bs x img_seq_lenght x d_model
        img_features = self.ffn(img_features + text_to_img_features) #bs x img_seq_lenght x d_model
        text_features = self.ffn(text_features + img_to_text_features) #bs x text_seq_lenght x d_model
        return img_features, text_features #bs x img_seq_lenght x d_model, #bs x text_seq_lenght x d_model

if __name__ == '__main__':
    # Example usage
    device='cpu'
    model = FeatureEnhancerLayer(1024)
    images = torch.randn(2, 3, 224, 224) # Example batch of images
    texts = ["a photo of a cat", "a picture is a dog"]  # Example batch of text descriptions
    bs = 2
    num_query = 4165
    num_levels = 4
    reference_points = torch.rand(bs, num_query, num_levels, 2)
    
    img_features, text_features = model(images, texts, reference_points)
    print(img_features.shape, text_features.shape)

