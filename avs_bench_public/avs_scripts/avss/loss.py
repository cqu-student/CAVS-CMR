import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np


def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total five frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1) # [bs*10, 224, 224]
    # loss = nn.CrossEntropyLoss()(pred_mask, ten_gt_masks)
    #! notice:
    loss = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(pred_mask, ten_gt_masks) # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1) # [bs*10]
    loss = loss * gt_temporal_mask_flag # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, threshold=False,\
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_AV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    bg_idx = 0
    pred_masks = torch.softmax(pred_masks, dim=1) # [B*10, NUM_CLASSES, 224, 224]
    pred_masks = torch.argmax(pred_masks, dim=1).unsqueeze(1) # [B*10, 1, 224, 224]
    pred_masks = (pred_masks != bg_idx).float() # [B*10, 1, 224, 224]
    total_loss = 0

    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*10, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        # downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        if threshold:
            downsample_pred_masks = (downsample_pred_masks > 0.5).float() # [bs*10, 1, H, W]
            obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*10, 1]
            masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*10, C, H, W]
            masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6)# [bs*10, C]
        else:
            masked_v_map = torch.mul(v_map, downsample_pred_masks)
            masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*10, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(a_fea, masked_v_fea, p=2) # [bs*10]
            # loss = euclidean_distance.mean()
            #! notice:
            loss = euclidean_distance * gt_temporal_mask_flag # [bs*10]
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        elif kl_flag:
            # loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='sum')
            #! notice:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='none') #[bs*10, C]
            loss = loss.sum(-1) # [bs*10]
            loss = loss * gt_temporal_mask_flag
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss


def closer_loss(pred_masks, a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, \
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_VV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    bg_idx = 0
    pred_masks = torch.softmax(pred_masks, dim=1) # [B*5, NUM_CLASSES, 224, 224]
    pred_masks = torch.argmax(pred_masks, dim=1).unsqueeze(1) # [B*5, 1, 224, 224]
    pred_masks = (pred_masks != bg_idx).float() # [B*5, 1, 224, 224]
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*5, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        # downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        ###############################################################################
        # pick the closest pair
        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)

        a_fea_simi = torch.cdist(a_fea,a_fea,p=2) # [BT, BT]
        a_fea_simi = a_fea_simi + 10*torch.eye(a_fea_simi.shape[0]).cuda() #
        idxs = a_fea_simi.argmin(dim=0) # [BT]

        masked_v_map = torch.mul(v_map, downsample_pred_masks)
        masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*10, C]
        if norm_fea:
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        target_fea = masked_v_fea[idxs]
        ###############################################################################
        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(target_fea, masked_v_fea, p=2)
            # loss = euclidean_distance.mean()
            #! notice:
            loss = euclidean_distance * gt_temporal_mask_flag # [bs*10]
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        elif kl_flag:
            # loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), target_fea.softmax(dim=-1), reduction='sum')
            #! notice:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='none') #[bs*10, C]
            loss = loss.sum(-1) # [bs*10]
            loss = loss * gt_temporal_mask_flag
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss

def BCEWithLogitsLossWithIgnoreIndex(inputs, targets, gt_temporal_mask_flag, reduction="mean", ignore_index=255):
    # inputs of size B x C x H x W
    n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
    labels_new = torch.where(targets != ignore_index, targets, n_cl)
    # replace ignore with numclasses + 1 (to enable one hot and then remove it)
    targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
    targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
    # targets is B x C x H x W so shape[1] is C
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    # loss has shape B x C x H x W
    #[20,71,244,244]
    zero_indices = torch.nonzero(gt_temporal_mask_flag == 0).squeeze()
    if zero_indices.numel() > 0:
        loss.index_put_(indices=[zero_indices], values=torch.zeros_like(loss[zero_indices]))
    loss = torch.sum(loss, dim=1) # sum the contributions of the classes    
    if reduction == 'mean':
        # if targets have only zeros, we skip them
        return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()/ torch.sum(gt_temporal_mask_flag) 
    elif reduction == 'sum':
        return torch.masked_select(loss, targets.sum(dim=1) != 0).sum() / torch.sum(gt_temporal_mask_flag) 
    else:
        return loss * targets.sum(dim=1)


def IcarlLoss(inputs, targets, output_old,gt_temporal_mask_flag, reduction='mean', ignore_index = 255, bkg = False):
    n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
    labels_new = torch.where(targets != ignore_index, targets, n_cl)
    # replace ignore with numclasses + 1 (to enable one hot and then remove it)
    targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
    targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
    if bkg:
        targets[:, 1:output_old.shape[1], :, :] = output_old[:, 1:, :, :]
    else:
        targets[:, :output_old.shape[1], :, :] = output_old
    # targets is B x C x H x W so shape[1] is C
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    # zero_indices = torch.nonzero(gt_temporal_mask_flag == 0).squeeze()
    # if zero_indices.numel() > 0:
    #     loss.index_put_(indices=[zero_indices], values=torch.zeros_like(loss[zero_indices]))
    # loss = torch.sum(loss, dim=1)
    # loss has shape B x C x H x W
      # sum the contributions of the classes
    if reduction == 'mean':
        # if targets have only zeros, we skip them
        return loss.mean() 
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def UnbiasedCrossEntropy(inputs, targets, gt_temporal_mask_flag, mask = None, old_cl=None, reduction='mean', ignore_index=255):
        old_cl = old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero
        if mask is not None:
            labels[mask] = ignore_index
        #TODO: 这里可能有问题，查看loss的shape
        loss = F.nll_loss(outputs, labels, ignore_index=ignore_index, reduction=reduction)
        loss = loss * gt_temporal_mask_flag
        loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        return loss
def KnowledgeDistillationLoss(inputs, targets, mask=None, reduction="mean", alpha=1.):
    inputs = inputs.narrow(1, 0, targets.shape[1])

    outputs = torch.log_softmax(inputs, dim=1)
    labels = torch.softmax(targets * alpha, dim=1)

    loss = (outputs * labels).mean(dim=1)

    if mask is not None:
        loss = loss * mask.float()

    if reduction == 'mean':
        outputs = -torch.mean(loss)
    elif reduction == 'sum':
        outputs = -torch.sum(loss)
    else:
        outputs = -loss

    return outputs
def safe_log(t,eps=1e-10):
    return torch.log(t+eps)

def FDA_loss(source_img, target_img, audio_feature, vid_temporal_mask_flag, model, old_model, L=0.1, lambda_kd=1.0):
    # 1. 风格迁移：混合频域特征
    mixed_img = FDA_source_to_target(source_img, target_img, L=L)
    
    # 2. 前向传播
    with torch.no_grad():
        old_output,v_map_list, a_fea_list, head, attentions = old_model(target_img, audio_feature, vid_temporal_mask_flag)  # 旧模型生成伪标签
    new_output,v_map_list, a_fea_list, head, attentions = model(mixed_img, audio_feature, vid_temporal_mask_flag)
    new_output = new_output[:, 0 : old_output.shape[1], :, :]  # 只保留旧模型的输出
    # 3. 知识蒸馏损失（KL散度）
    kd_loss = F.kl_div(
        F.log_softmax(new_output, dim=1),
        F.softmax(old_output, dim=1),
        reduction='batchmean'
    )
    
    # 4. 总损失
    total_loss = lambda_kd * kd_loss
    return total_loss

def FDA_source_to_target(img_src, img_trg, L=0.1):
    def low_freq_mutate(amp_src, amp_trg):
        b, c, h, w = amp_src.shape
        boundary = int(min(h, w) * L)
        mask = torch.ones_like(amp_src)
        for y in range(h):
            for x in range(w):
                if (x - w//2)**2 + (y - h//2)**2 > boundary**2:
                    mask[:, :, y, x] = 0
        return amp_src * mask + amp_trg * (1 - mask)
    
    fft_src = torch.fft.fft2(img_src, dim=(-2, -1))
    fft_trg = torch.fft.fft2(img_trg, dim=(-2, -1))
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg, pha_trg = torch.abs(fft_trg), torch.angle(fft_trg)
    
    amp_src = low_freq_mutate(amp_src, amp_trg)
    fft_mix = torch.polar(amp_src, pha_src)
    img_mix = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real
    return img_mix

class Distillation_loss_unbiased(nn.Module):
    def __init__(self, unknown_label=-1, device='cuda'):
        super().__init__()
        self.unknown_label = unknown_label
        self.device = device

        self.old_classes, self.new_classes = [], []

    def update_classes(self, num_classes):
        self.old_classes += self.new_classes
        self.new_classes = [i for i in range(max(self.old_classes)+1 if self.old_classes else 0, num_classes)]
        if self.unknown_label in self.new_classes:
            self.new_classes.remove(self.unknown_label)

    def forward(self, pred_new, pred_old, mask=None):

        softmax_old = F.softmax(pred_old.detach().clone(), dim=1)
        softmax_new = F.softmax(pred_new, dim=1)
        # merge unknown and new classes to unknown class to match label distribution of old prediction
        classes_to_merge = torch.Tensor(self.new_classes + [self.unknown_label]).long().to(self.device)
        classes_to_keep = torch.Tensor(self.old_classes).long().to(self.device)
        softmax_new_new = softmax_new.index_select(dim=1, index=classes_to_merge).sum(dim=1).unsqueeze(1)
        softmax_new_old = softmax_new.index_select(dim=1, index=classes_to_keep)

        log_softmax_new = safe_log(torch.cat([softmax_new_new,softmax_new_old], dim=1))
        loss = softmax_old * log_softmax_new
        if mask is not None:
            loss = loss * mask.float()
        loss = -1 * torch.mean(loss)
        return loss

def soft_crossentropy(logits, labels, logits_old, mask_valid_pseudo,
                      mask_background, pseudo_soft, pseudo_soft_factor=1.0):
    if pseudo_soft not in ("soft_certain", "soft_uncertain"):
        raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
    nb_old_classes = logits_old.shape[1]
    bs, nb_new_classes, w, h = logits.shape

    loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    loss_uncertain = (torch.log_softmax(logits_old, dim=1) * torch.softmax(logits[:, :nb_old_classes], dim=1)).sum(dim=1)

    if pseudo_soft == "soft_certain":
        mask_certain = ~mask_background
        mask_uncertain = mask_valid_pseudo & mask_background
    elif pseudo_soft == "soft_uncertain":
        mask_certain = (mask_valid_pseudo & mask_background) | (~mask_background)
        mask_uncertain = ~mask_valid_pseudo & mask_background

    loss_certain = mask_certain.float() * loss_certain
    loss_uncertain = (~mask_certain).float() * loss_uncertain

    return loss_certain + pseudo_soft_factor * loss_uncertain
def nca(
    similarities,
    targets,
    loss,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    b = similarities.shape[0]
    c = similarities.shape[1]
    w = similarities.shape[-1]

    if margin > 0.:
        similarities = similarities.view(b, c, w * w)
        targets = targets.view(b * w * w)
        margins = torch.zeros_like(similarities)
        margins = margins.permute(0, 2, 1)
        margins[torch.arange(margins.shape[0]), targets, :] = margin
        margins = margins.permute(0, 2, 1)
        similarities = similarities - margin
        similarities = similarities.view(b, c, w, w)
        targets = targets.view(b, w, w)

    similarities = scale * similarities

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(dim=1, keepdims=True)[0]  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return loss(similarities, targets)


def NCA(inputs, targets, scale = 1., margin=0., ignore_index = 255, reduction="mean"):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    return nca(inputs, targets, ce, scale, margin)

def UnbiasedNCA(inputs, targets, scale = 1., margin = 0., old_cl = None, reduction="mean", ignore_index = 255):
    unce = UnbiasedCrossEntropy(old_cl, reduction, ignore_index)
    return nca(inputs, targets, unce, scale=scale, margin=margin)

def UnbiasedKnowledgeDistillationLoss(inputs, targets, mask=None, reduction="mean", alpha=1.):
    new_cl = inputs.shape[1] - targets.shape[1]

    targets = targets * alpha

    new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

    den = torch.logsumexp(inputs, dim=1)                          # B, H, W
    outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
    outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

    labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

    # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
    loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

    if mask is not None:
        loss = loss * mask.float()

    if reduction == 'mean':
        outputs = -torch.mean(loss)
    elif reduction == 'sum':
        outputs = -torch.sum(loss)
    else:
        outputs = -loss

    return outputs

def IouSemanticAwareLoss(pred_masks, gt_mask, \
                        a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, scale, mask_background, mask_valid_pseudo, pseudo_labels,\
                        sa_loss_flag=False, count_stages=[], lambda_1=0, \
                        mask_pooling_type='avg', norm_fea=True, \
                        threshold=False, closer_flag=False, euclidean_flag=False, kl_flag=False, cfg=None, classif_adaptive_factor = 1.):
    """
    loss for multiple sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    gt_mask: ground truth mask of the first frame (one-shot) or five frames, shape: [bs*10, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    total_loss = 0
    if cfg.icarl:
        iou_loss = BCEWithLogitsLossWithIgnoreIndex(pred_masks, gt_mask, gt_temporal_mask_flag)
    elif cfg.unce and cfg.step!=0:
        iou_loss = UnbiasedCrossEntropy(pred_masks, gt_mask, gt_temporal_mask_flag, old_cl=sum(cfg.classes), ignore_index=255)
    elif cfg.nca and cfg.old_classes !=0:
        pdb.set_trace()
        _labels = gt_mask.clone()
        _labels[~(mask_background & mask_valid_pseudo)] = 255
        _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &mask_valid_pseudo]
        loss_not_pseudo = UnbiasedNCA(pred_masks, gt_mask, mask_background, cfg.pseudo_soft, pseudo_soft_factor=cfg.pseudo_soft_factor)
        loss_pseudo = F.cross_entropy(pred_masks, _labels, ignore_index=255, reduction="none")
        iou_loss = loss_pseudo + loss_not_pseudo
    elif cfg.nca:
        iou_loss = NCA(pred_masks, gt_mask, scale=scale, margin=cfg.nca_margin, ignore_index = 255, reduction = "none")
    elif cfg.ce_on_new:
        _labels = gt_mask.clone()
        _labels[_labels == 0] = 255
        iou_loss = F10_IoU_BCELoss(pred_masks, _labels, gt_temporal_mask_flag)
    else:
        iou_loss = F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss
    sa_loss_flag = False
    if sa_loss_flag:
        if closer_flag: # Loss_AVM_VV reported in the paper
            masked_av_loss = closer_loss(pred_masks, a_fea_list, v_map_list, gt_temporal_mask_flag, count_stages, mask_pooling_type, norm_fea, euclidean_flag, kl_flag)
        else: # Loss_AVM_AV reported in the paper
            masked_av_loss = A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, gt_temporal_mask_flag, count_stages, mask_pooling_type, norm_fea, threshold, euclidean_flag, kl_flag)
        total_loss += lambda_1 * masked_av_loss
    else:
        masked_av_loss = torch.zeros(1)
    sample_weights = None
    if cfg.sample_weights_new is not None:
        sample_weights = torch.ones_like(gt_mask).to(gt_mask.device, dtype = torch.float32)
        sample_weights[gt_mask >= 0] = cfg.sample_weights_new
    if sample_weights is not None:
        iou_loss = iou_loss * sample_weights
    iou_loss = classif_adaptive_factor * iou_loss
    iou_loss = iou_loss.mean()
    loss_dict = {}
    loss_dict['iou_loss'] = iou_loss.item()
    loss_dict['sa_loss'] = masked_av_loss.item()
    loss_dict['lambda_1'] = lambda_1

    return total_loss, loss_dict


if __name__ == "__main__":

    pdb.set_trace()
