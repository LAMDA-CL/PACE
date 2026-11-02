import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import PaceNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import math
import matplotlib.pyplot as plt
import os
import json
import random

random.seed(1993)
np.random.seed(1993)

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._train_transformer = True
        self._network = PaceNet(args)

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self._known_classes = 0
        self.prototype = []
        self.pat_pro = []
        self.embedding_dim = get_attribute(args, "embedding_dim", 512)
        self.mean = []
        self.cov = []
        self.inter_mean = []
        self.inter_cov = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.cross_projector.store_old_subspace()

    def _get_batch_des(self, des_file, classnames):
        batch_des = []
        for classname in classnames:
            batch_des.append(classname + ' with ' + random.choice(des_file[classname]).lower())
        return batch_des

    @torch.no_grad()
    def global_to_local(self, gloal_features, local_features,return_indices=False):
        #global[b,d]，local[b,p,d]
       # gloal_features, local_features = self._network.visual_forward_tokens(image)
        gloal_features = gloal_features / gloal_features.norm(dim=-1, keepdim=True)
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)

        image_fine_list = []
        top_fine_list = []
        B, d = gloal_features.shape

        _, _, before_d = local_features.shape
        x = local_features  # B,N+1,d
        x = x.reshape(-1, before_d)  #

        #  x = self._network.mid_image_trans(x)
        x = x.reshape(B, -1, d)  # B,N+1,d

        local_feature = x
        image_fine_list.append(local_feature)
        patches = x[:, 1:, :]  # [B, N, d]
        patches_norm = patches / (patches.norm(dim=-1, keepdim=True))
        sim = (patches_norm * gloal_features.unsqueeze(1)).sum(dim=-1)

        k = self.args["top_k"]
        _, indices = torch.topk(sim, k=k, dim=-1)  # （64，5）   the size of batch is 64
        indices = torch.cat((torch.zeros(B, 1, dtype=torch.int64).cuda(), indices), dim=1)  # (64,6)

        top_fine_feature = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(B, k + 1, d))

        avg_feature = torch.mean(x, dim=1, keepdim=True)
        top_fine_feature = torch.cat((top_fine_feature, avg_feature), dim=1)
        top_fine_list.append(top_fine_feature.reshape(B,-1, d))

        if len(image_fine_list) > 0:
            image_fine_list = torch.cat(image_fine_list)
            top_fine_list = torch.cat(top_fine_list)
        if return_indices:
            return gloal_features, top_fine_list, indices.cpu().numpy()
        else:
            return gloal_features, top_fine_list


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
      #  self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.update_task()
        self._network.update_stat(self._known_classes, self._total_classes, self.train_loader, self._device)
        self.train(self.train_loader, self.test_loader, train_dataset)

    def train(self, train_loader, test_loader, train_dataset):
        self._network.eval()
        # analyze
        sample_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
        x = []
        y = []
        p = []
        z = []
        for i, (_, inputs, targets) in enumerate(sample_loader):
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fea = self._network.model.encode_image(inputs)
                _, fine_fea = self._network.visual_forward_tokens(inputs)  # [b,p,d]
                # print(fine_fea.shape) #
                # _, fine = self._network.visual_forward_tokens(x)
                x.append(fea.cpu())
                y.append(targets.cpu())
                p.append(fine_fea.cpu())

        y = torch.cat(y, dim=0)
        x = torch.cat(x, dim=0)
        p = torch.cat(p, dim=0)
        # z = torch.cat(z, dim=0)
        label = torch.sort(torch.unique(y))[0]
        for l in label:
            index = torch.nonzero(y == l).squeeze()
            l_fea = x[index]
            l_mean = l_fea.mean(dim=0)
            l_fine = p[index]
            fine_mean = l_fine.mean(dim=0)
            self.prototype.append(l_mean.to(self._device))
            self.pat_pro.append(fine_mean.to(self._device))
        self._network.eval()
        params = list()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                params.append((name, param))
                print(f'{name}')
        self._network.to(self._device)
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                  weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        prog_bar = tqdm(range(self.tuned_epoch))
        total_labels = class_to_label[:self._total_classes]
        templates = self.data_manager._data_to_prompt[0]

        from utils.toolkit import ClipLoss
        cliploss = ClipLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            correct_clip = []
            if self._cur_task > 0:
                old_class = list(range(self.args['init_cls'] + (self._cur_task - 1) * self.args['increment']))
                random.shuffle(old_class)
            for i, (m, inputs, targets) in enumerate(train_loader):
               # print(m)
                inputs = inputs.to(self._device)
                # print(inputs.shape)
                targets = targets.to(self._device)
                if self._cur_task > 0:
                    proto_x = []
                    proto_y = []
                    proto_z = []
                    old_class = random.choices(old_class, k=self.args['sample_num']) if self.args[
                                                                                            'sample_num'] > 0 else old_class

                    for j in old_class:
                        proto_x.append(torch.stack(
                            [self.prototype[j] + torch.randn_like(self.prototype[j]) * self.args['sample_noise']],
                            dim=0))
                        proto_z.append(torch.stack(
                            [self.pat_pro[j] + torch.randn_like(self.pat_pro[j]) * self.args['sample_noise']], dim=0))
                        proto_y.append(torch.ones(int(1), dtype=torch.long, device=self._device) * j)
                    proto_x = torch.cat(proto_x, dim=0)

                    proto_y = torch.cat(proto_y, dim=0)
                    proto_z = torch.cat(proto_z, dim=0)
                    targets = torch.cat([targets, proto_y], dim=0)

                gloabl_img, patch_img = self._network.visual_forward_tokens(inputs)
                gloabl_img, patch_img = self.global_to_local(gloabl_img,patch_img)
              #  print(gloabl_img.shape,patch_img.shape)

                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                gloabl_text, patch_text = self._network.text_forward_tokens(texts)
                gloabl_text, patch_text = self.global_to_local(gloabl_text, patch_text)

                from utils.loss import CosineSimilarityLoss, InfoNCELoss, contrastive_loss
                image_features, _ = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                if self._cur_task > 0:
                    sg_image_features = self._network.Image_encode(proto_x)
                    sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
                    img_feas = torch.cat([img_feas, sg_image_features], dim=0)
                    sg_img,sg_fine=self.global_to_local(proto_x, proto_z)
                    gloabl_img =torch.cat([gloabl_img, sg_img], dim=0)
                    patch_img = torch.cat([patch_img, sg_fine], dim=0)


                img_g, text_l, logit_scale = self._network.forward_transformer(gloabl_img, patch_text, self._train_transformer)
                text_g, _, logit_scale = self._network.forward_transformer(gloabl_text,patch_img,
                                                                               self._train_transformer)
                img_l,_,logit_scale =self._network.forward_transformer(patch_img, gloabl_img, self._train_transformer)
                text_l,_,logit_scale =self._network.forward_transformer(patch_text,gloabl_text, self._train_transformer)
                b,_ ,d = img_g.shape
                img_g=img_g.view(b,d)
                top_image_fine = img_l.view(b, -1, d)  # [B, K, D]
                query_feats = top_image_fine.mean(dim=1)  # [B, D]
                query_feats = self._network.Image_encode(query_feats)
                query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
                c,_,d =text_g.shape
                text_g = text_g.view(c,d)
                text_fine = text_l.view(c, -1, d)
                text_feats = text_fine.mean(dim=1)  # [B, D]
                text_feats = self._network.Text_encode(text_feats)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                img_g = self._network.Image_encode(img_g)
                img_g = img_g/img_g.norm(dim=-1, keepdim=True)
                text_g = self._network.Text_encode(text_g)
                text_g = text_g/text_g.norm(dim=-1, keepdim=True)
                logit_scale = self._network.model.logit_scale
                logits1 = img_g @ text_g.T
                logits2 = query_feats @ text_feats.T
                loss1 = F.cross_entropy(logits1, targets)
              #  loss2 = F.cross_entropy(logits2, targets)

                global_logits = img_feas @ text_feas.T
                labels = [class_to_label[y] for y in targets]
                texts_clip = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(self._network.tokenizer(texts_clip).to(self._device))
                clip_text_norm = clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm
                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                logits = self.args["fusion_ratio"] * global_logits + (1 - self.args["fusion_ratio"]) * (logits1)
                loss = clip_loss + loss1*self.args["alpha"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                correct_clip.append(preds.eq(targets).cpu())
                total += len(targets)
                if (epoch + 1) == self.tuned_epoch and self._cur_task > 0:
                    self._network.cross_projector.project_new_weights()

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader, epoch)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], losses / len(train_loader), train_acc, test_acc, )
            prog_bar.set_description(info)


    def _compute_accuracy(self, model, loader, epoch=0):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        correct, total = 0, 0
        for i, (m, inputs, targets) in enumerate(loader):
            #print(m)
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # image_features = self._network.model.encode_image(inputs)
                # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # protos = torch.stack(self.prototype, dim=0)
                # sim_logits = image_features @ protos.T
                transf_image_features, _ = self._network.encode_image(inputs)
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                transf_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                gloabl_img, patch_img = self._network.visual_forward_tokens(inputs)
                gloabl_img, patch_img = self.global_to_local(gloabl_img, patch_img)
                #  print(gloabl_img.shape,patch_img.shape)
                templates = self.data_manager._data_to_prompt[0]

                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                # text_features = self._network.encode_text(texts)
                # text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                gloabl_text, patch_text = self._network.text_forward_tokens(texts)
                gloabl_text, patch_text = self.global_to_local(gloabl_text, patch_text)
                img_g, text_l, logit_scale = self._network.forward_transformer(gloabl_img, patch_text,
                                                                               self._train_transformer)
                #  print(img_g.shape, text_l.shape, logit_scale)
                text_g, img_l, logit_scale = self._network.forward_transformer(gloabl_text, patch_img,
                                                                               self._train_transformer)

                b, _, d = img_g.shape
                img_g = img_g.view(b, d)
                top_image_fine = img_l.view(b, -1, d)  # [B, K, D]
                # print(top_image_fine.shape)
                query_feats = top_image_fine.mean(dim=1)  # [B, D]
                query_feats = self._network.Image_encode(query_feats)
                query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
                c, _, d = text_g.shape
                text_g = text_g.view(c, d)
                text_fine = text_l.view(c, -1, d)
                text_feats = text_fine.mean(dim=1)  # [B, D]
                text_feats = self._network.Text_encode(text_feats)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                img_g = self._network.Image_encode(img_g)
                img_g = img_g / img_g.norm(dim=-1, keepdim=True)
                text_g = self._network.Text_encode(text_g)
                text_g = text_g / text_g.norm(dim=-1, keepdim=True)
                logits1 = img_g @ text_g.T
                logits2 = query_feats @ text_feats.T
                if epoch == self.args['tuned_epoch']:
                    transf_image_features_raw_ = self._network.visual_forward_(inputs)
                    transf_image_features_raw = transf_image_features_raw_ @ self._network.visual_proj
                    transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1,
                                                                                                              keepdim=True)
                    transf_image_features_raw = transf_image_features_raw / transf_image_features_raw.norm(dim=-1,
                                                                                                           keepdim=True)
                    outputs = (transf_image_features @ transf_text_features.T)
                    outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
                    outputs_gda = outputs_gda / outputs_gda.norm(dim=-1, keepdim=True)
                    outputs = outputs_gda * self.args['stat']  + outputs * ( 1 - self.args['stat'])
                else:
                    outputs = (transf_image_features @ transf_text_features.T)
                    outputs = (transf_image_features @ transf_text_features.T)
                   # print(outputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        text_features = text_features.to(self._device)
        transf_text_features = text_features

        y_pred, y_true,y_orn = [], [],[]
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # image_features =self._network.model.encode_image(inputs)
                # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # protos = torch.stack(self.prototype, dim=0)
                # sim_logits = image_features @ protos.T
                transf_image_features, _ = self._network.encode_image(inputs)
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                transf_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                gloabl_img, patch_img = self._network.visual_forward_tokens(inputs)
                gloabl_img, patch_img = self.global_to_local(gloabl_img, patch_img)
                #  print(gloabl_img.shape,patch_img.shape)
                templates = self.data_manager._data_to_prompt[0]
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                # text_features = self._network.encode_text(texts)
                # text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                gloabl_text, patch_text = self._network.text_forward_tokens(texts)
                gloabl_text, patch_text = self.global_to_local(gloabl_text, patch_text)
                img_g, text_g, logit_scale = self._network.forward_transformer(gloabl_img, gloabl_text,
                                                                               self._train_transformer)
                #  print(img_g.shape, text_l.shape, logit_scale)
                img_l, text_l, logit_scale = self._network.forward_transformer(patch_img, patch_text,
                                                                               self._train_transformer)
                b, _, d = img_g.shape
                img_g = img_g.view(b, d)
                top_image_fine = img_l.view(b, -1, d)  # [B, K, D]
                # print(top_image_fine.shape)
                query_feats = top_image_fine.mean(dim=1)  # [B, D]
                query_feats = self._network.Image_encode(query_feats)
                query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
                c, _, d = text_g.shape
                text_g = text_g.view(c, d)
                text_fine = text_l.view(c, -1, d)
                text_feats = text_fine.mean(dim=1)  # [B, D]
                text_feats = self._network.Text_encode(text_feats)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                img_g = self._network.Image_encode(img_g)
                img_g = img_g / img_g.norm(dim=-1, keepdim=True)
                text_g = self._network.Text_encode(text_g)
                text_g = text_g / text_g.norm(dim=-1, keepdim=True)
                logits1 = img_g @ text_g.T
                logits2 = query_feats @ text_feats.T


                output = (transf_image_features @ transf_text_features.T) * self.args["fusion_ratio"] + (
                            1 - self.args["fusion_ratio"]) * (logits1)

                transf_image_features_raw_ = self._network.visual_forward_(inputs)
                transf_image_features_raw = transf_image_features_raw_ @ self._network.visual_proj
                transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1, keepdim=True)
                outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
                outputs_gda = outputs_gda / outputs_gda.norm(dim=-1, keepdim=True)
                outputs = outputs_gda* self.args['stat'] + output * (1 - self.args['stat'])

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            pre = torch.topk(output, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        #    y_orn.append(pre.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)# [N, topk]


