import torch 
import numpy as np 
import torch.nn as nn
from torchvision import models

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

convnext_dict = {"convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "convnext_large": "convnext_large"}

class ConvNextBase(nn.Module):
    def __init__(self, model_name="convnext_base"):
        super(ConvNextBase, self).__init__()
        

        if model_name not in convnext_dict:
            raise ValueError(f"不支持的ConvNeXt模型: {model_name}。可选: {list(convnext_dict.keys())}")

        weights = {
            "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            "convnext_small": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
            "convnext_base": models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
            "convnext_large": models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        }[model_name]
        
        model_convnext = models.get_model(
            convnext_dict[model_name], weights=weights
        )
        
        self.features = model_convnext.features
        self.avgpool = model_convnext.avgpool
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.forward(dummy_input)
            self.backbone_feat_dim = dummy_output.shape[1]
            
            del dummy_input, dummy_output
            torch.cuda.empty_cache()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, 
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
    

class SFUniDA(nn.Module):
    def __init__(self, args):
        super(SFUniDA, self).__init__()
        self.backbone_arch = args.backbone_arch   
        self.embed_feat_dim = args.embed_feat_dim 
        self.class_num = args.class_num           
        if "resnet" in self.backbone_arch:   
            self.backbone_layer = ResBase(self.backbone_arch) 
        elif "convnext" in self.backbone_arch:
            self.backbone_layer = ConvNextBase(self.backbone_arch)
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")  
        

    def forward(self, input_imgs, apply_softmax=True):
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        cls_out = self.class_layer(embed_feat)
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1) 
        
        return embed_feat, cls_out
