import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import timm


class TextBackbone(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(TextBackbone, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state   # [CLS] token representation
        return pooled_output

class ImageBackbone(nn.Module):
    def __init__(self, pretrained_model_name='swin_base_patch4_window7_224'):
        super(ImageBackbone, self).__init__()
        self.swin = timm.create_model(pretrained_model_name, features_only=True, pretrained=True)
        #self.swin.reset_classifier(0)  # Remove the original classifier
    
    def forward(self, images):
        
        features = self.swin(images)
        # features = features.flatten(start_dim=1, end_dim=2)
        return features

class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_model_name='swin_base_patch4_window7_224'):
        super(MultiModalFeatureExtractor, self).__init__()
        self.text_backbone = TextBackbone(pretrained_model_name=text_model_name)
        self.image_backbone = ImageBackbone(pretrained_model_name=image_model_name)

    def forward(self, texts, images):
        text_features = self.text_backbone(texts)
        image_features = self.image_backbone(images)
        return text_features, image_features