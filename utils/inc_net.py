import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear, CosineLinear_RanPAC
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
from typing import Optional
import os
import json
import torchvision.transforms as transforms
from utils.toolkit import get_attribute
import difflib
from PIL import Image
import random
import math
random.seed(1993)

def get_convnet(args, pretrained=False):

    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()

    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=False)
            # model.load_state_dict(torch.load('c.pth'))
            # tokenizer = open_clip.get_tokenizer('ViT-B-16')
            # model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim=512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

from torch import Tensor
class Linear(nn.Module):
    # def __init__(self, in_dim: int, identity_init: bool = True, bias=False) -> NoneType:
    def __init__(self, in_dim: int, identity_init: bool = True, bias=False) :

        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=bias)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class PaceNet(BaseNet):
    def __init__(self, args, pretrained=None):
        super().__init__(args, pretrained)
        self.model, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.visual = self.model.visual
        self.visual_proj = self.visual.proj
        # self.transform = self.model.transform
        self.token_embedding = self.model.token_embedding
        self.positional_embedding = self.model.positional_embedding
        self.attn_mask = self.model.attn_mask
        self.ln_final =self.model.ln_final
        self.args=args
        self.freeze(self.model)
        self.Image_Adapter = nn.ModuleList()
        self.Text_Adapter = nn.ModuleList()
        self.beta = 1
        self.decay = 1
        device = args["device"][0]
        self.device = device
        self.class_mean_list = []
        self.class_cov_list = []
        self.class_edge_distance = []
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        from utils.proj import CrossAttentionProjector
        self.cross_projector = CrossAttentionProjector(
            self.sel_attn,
            alpha=self.args.get("svd_alpha", 0.9),
            threshold=self.args.get("svd_threshold", 0.95),
            device=self.device
        )
        
    def update_stat(self, known_classes, total_classes, train_loader,device):
        print("updating stat")
        with torch.no_grad():
            vecs = []
            # vecs_512 = []
            labels = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                image_features= self.visual_forward_(inputs)
                # image_features_512 = image_features @ self.visual_proj
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                vecs.append(image_features)
                # vecs_512.append(image_features_512)
                labels.append(targets)

            vecs = torch.cat(vecs)
            # vecs_512 = torch.cat(vecs_512)
            labels = torch.cat(labels)
            
            mu = torch.cat([vecs[labels == i].mean(dim=0, keepdim=True) for i in range(known_classes, total_classes)], dim=0)
            center_vecs = torch.cat([vecs[labels == i] - mu[i - known_classes] for i in range(known_classes, total_classes)], dim=0)
            cov_inv = center_vecs.T @ center_vecs /(center_vecs.shape[0] - 1)
            cov_inv =  center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())    
            if not hasattr(self, 'mu'):
                self.mu = mu
                self.cov_inv = cov_inv
            else:
                self.cov_inv = (known_classes/total_classes)*self.cov_inv + (total_classes-known_classes)/total_classes*cov_inv + ((known_classes/total_classes)*(total_classes-known_classes)/total_classes**2)*(self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)) @ (self.mu.T.mean(dim=1).unsqueeze(1) - mu.T.mean(dim=1).unsqueeze(1)).T
                self.mu = torch.cat([self.mu, mu])
            ps = torch.ones(self.mu.shape[0]).cuda() * 1. / self.mu.shape[0]
            self.W = torch.einsum('nd, dc -> cn', self.mu, self.cov_inv)
            self.b = ps.log() - torch.einsum('nd, dc, nc -> n', self.mu, self.cov_inv, self.mu) / 2


    def _expand_token(self, token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def forward_transformer(self, image_features, text_features, transformer=False):
        if transformer:
            len_texts = text_features.shape[0]
            # restack the features and pass them through the attention layer
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim) #[bs, 1, dim]/[bs, k, dim]
            len_image = image_features.shape[0]
            text_features = text_features.view(-1, self.feature_dim) #[total_classes*k, dim]
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim) #[bs, total_classes*k, dim]

            features = torch.cat([image_features, text_features], dim=1) # bsize * (1+total_classes*k) * dim
          #  features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)
            # pass through the attention layer
            features = self.sel_attn(image_features, text_features, text_features)
            # split them back, image features are the first half, text features are the second half
            # image_features, text_features = torch.split(features, features.shape[1] // 2, dim=1)
          #   image_features = features[:, :len_image, :] # bsize * dim
          # #  print(image_features.shape)
          #   text_features = features[:, len_image:, :] # bsize * num_texts * dim
          #   # remove the 0-th dimension of text features to be num_texts * dim
            text_features = torch.mean(text_features, dim=0) # num_texts * dim
            # squeeze
            image_features = features.view(image_features.shape[0], -1,self.feature_dim) #[bs, 1, dim]/[bs, k, dim]
            text_features = text_features.view(text_features.shape[0],-1, self.feature_dim)  #
            return image_features, text_features, self.model.logit_scale.exp()
        else:
            return image_features, text_features, self.model.logit_scale.exp()

    def text_global_pool(self,x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            assert text is not None
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    def text_forward_tokens(self, text):
        cast_dtype = self.model.transformer.get_cast_dtype()

        x = self.model.token_embedding(text).to(cast_dtype)
        x = x + self.model.positional_embedding[: x.size(1)].to(cast_dtype)

        x = self.model.transformer(x, attn_mask=self.model.attn_mask)

        tokens_last = self.model.ln_final(x)  # [B, L, width]

        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                tokens_out = self.model.text_projection(tokens_last)  # [B, L, D]
            else:
                tokens_out = tokens_last @ self.model.text_projection  # [B, L, D]
        else:
            tokens_out = tokens_last
        pooled, _ = self.text_global_pool(tokens_last, text, self.model.text_pool_type)
       # print(self.model.text_pool_type)
        if self.model.text_projection is not None:
            if isinstance(self.model.text_projection, nn.Linear):
                pooled = self.model.text_projection(pooled)
            else:
                pooled = pooled @ self.model.text_projection
        pooled_token = pooled.unsqueeze(1)  # [B, 1, D]
        tokens_out = torch.cat([pooled_token,
                                tokens_out], dim=1)
        return pooled, tokens_out

    def visual_forward_tokens(self, x):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x],
            dim=1
        )
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD → LND

        for blk in self.visual.transformer.resblocks:
            x = blk(x)

        tokens_last = x.permute(1, 0, 2)  # LND → NLD: [B, N+1, C]

        tokens_norm = self.visual.ln_post(tokens_last)

        if self.visual.proj is not None:
            tokens_out = tokens_norm @ self.visual.proj  # [B, N+1, D_out]
        else:
            tokens_out = tokens_norm

        # final CLS embedding
        final_cls = tokens_out[:, 0, :]

        return final_cls, tokens_out

    def visual_forward_(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([self._expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)

        if self.visual.attn_pool is not None:
            if self.visual.attn_pool_contrastive is not None:
                x = self.visual.ln_post(x)
                tokens = self.visual.attn_pool(x)
                if self.visual.attn_pool_type == 'parallel':
                    pooled = self.visual.attn_pool_contrastive(x)
                else:
                    assert self.visual.attn_pool_type == 'cascade'
                    pooled = self.visual.attn_pool_contrastive(tokens)
            else:
                x = self.visual.attn_pool(x)
                x = self.visual.ln_post(x)
                pooled, tokens = self.visual._global_pool(x)
        elif self.visual.final_ln_after_pool:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)
        else:
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        return pooled

    def update_task(self):
        from convs.linears import Adapter,MLP_Adapter
        if len(self.Image_Adapter)>0:
            self.freeze(self.Image_Adapter[-1])
            self.freeze(self.Text_Adapter[-1])
        self.Image_Adapter.append(MLP_Adapter(512, 512))
        self.Text_Adapter.append(MLP_Adapter(512, 512))
    
    def Image_encode(self, image_features):
        image_res = []
        for i in range(len(self.Image_Adapter)):
            image_res.append(self.Image_Adapter[i](image_features))
        image_res = torch.sum(torch.stack(image_res), dim=0)
        return image_res
    
    def Text_encode(self, text_features):
        text_res = []
        for i in range(len(self.Text_Adapter)):
            text_res.append(self.Text_Adapter[i](text_features))
        text_res = torch.sum(torch.stack(text_res), dim=0)
        return text_res

    @property
    def feature_dim(self):
        return self.model.out_dim
    
    def extract_vector(self, x):
        return self.model.encode_image(x)

    def encode_image(self, x):
        imag_features =  self.model.encode_image(x)
       # print(imag_features.shape)
        imag_fea , fine =self.visual_forward_tokens(x)
      #  print(imag_features-imag_features)
      #  print(imag_features.shape,fine.size())
        imag_res = self.Image_encode(imag_features)
        B, L, D = fine.shape  # L = N+1
        fine_flat = fine.reshape(B * L, D)  # [B*L, D]

        fine_out_flat = self.Image_encode(fine_flat)  # [B*L, d_out]

        fine_out = fine_out_flat.reshape(B, L, -1)  # [B, L, d_out]

        return imag_res, fine_out
    
    def encode_text(self, x):
        text_features = self.model.encode_text(x)
        text_res = self.Text_encode(text_features)
        return text_res

    def forward(self, img, text):
        image_features, text_features, logit_scale=self.model(img, text)
        return image_features, text_features, logit_scale
        
    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def activate_old_adapter(self):
        for item in self.Image_Adapter:
            for param in item.parameters():
                param.requires_grad = True

        for item in self.Text_Adapter:
            for param in item.parameters():
                param.requires_grad = True

