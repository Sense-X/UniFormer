import numpy as np
import torch
from torchvision.ops import roi_align

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def get_featuremaps(label_maps_topk, num_classes, device='cuda'):
    label_maps_topk_sizes = label_maps_topk[0].size()
    label_maps = torch.full([label_maps_topk.size(0), num_classes, label_maps_topk_sizes[2],
                              label_maps_topk_sizes[3]], 0, dtype=torch.float32 ,device=device)
    for _label_map, _label_topk in zip(label_maps, label_maps_topk):
        _label_map = _label_map.scatter_(
            0,
            _label_topk[1][:, :, :].long(),
            _label_topk[0][:, :, :].float()
        )
    return label_maps

def get_label(label_maps, batch_coords,label_size=1,device='cuda'):
    '''
    Adapted from https://github.com/naver-ai/relabel_imagenet/blob/main/utils/relabel_functions.py
    Here we generate label for patch tokens and cls token separately and concat them together if given label_size>1
    '''
    num_batches = label_maps.size(0)
    target_label = roi_align(
        input=label_maps,
        boxes=torch.cat(
            [torch.arange(num_batches).view(num_batches,
                                            1).float().to(device),
             batch_coords.float() * label_maps.size(3) - 0.5], 1),
        output_size=(label_size, label_size))
    if label_size>1:
        target_label_cls = roi_align(
            input=label_maps,
            boxes=torch.cat(
                [torch.arange(num_batches).view(num_batches,
                                                1).float().to(device),
                 batch_coords.float() * label_maps.size(3) - 0.5], 1),
            output_size=(1, 1))
        B,C,H,W = target_label.shape
        target_label = target_label.view(B,C,H*W)
        target_label = torch.cat([target_label_cls.view(B,C,1),target_label],dim=2)
    target_label = torch.nn.functional.softmax(target_label.squeeze(), 1)
    return target_label

def get_labelmaps_with_coords(label_maps_topk, num_classes, on_value=1., off_value=0.,label_size=1, device='cuda'):
    '''
    Adapted from https://github.com/naver-ai/relabel_imagenet/blob/main/utils/relabel_functions.py
    Generate the target label map for training from the given bbox and raw label map
    '''
    # trick to get coords_map from label_map
    random_crop_coords = label_maps_topk[:,2,0,0,:4].view(-1, 4)
    random_crop_coords[:, 2:] += random_crop_coords[:, :2]
    random_crop_coords = random_crop_coords.to(device)

    # trick to get ground truth from label_map
    ground_truth = label_maps_topk[:,2,0,0,5].view(-1).to(dtype=torch.int64)
    ground_truth = one_hot(ground_truth, num_classes, on_value=on_value, off_value=off_value, device=device)

    # get full label maps from raw topk labels
    label_maps = get_featuremaps(label_maps_topk=label_maps_topk,
                               num_classes=num_classes,device=device)

    # get token-level label and ground truth
    token_label = get_label(label_maps=label_maps,
                          batch_coords=random_crop_coords,
                          label_size=label_size,
                          device=device)
    B,C = token_label.shape[:2]
    token_label = token_label*on_value+off_value
    if label_size==1:
        return torch.cat([ground_truth.view(B,C,1),token_label.view(B,C,1)],dim=2)
    else:
        return torch.cat([ground_truth.view(B,C,1),token_label],dim=2)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda', label_size=1):
    '''
    generate and mix target from the given label maps
    target: label maps/ label maps with coords 
    num_classes: number of classes for the target
    lam: lambda for mixup target
    ''' 
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    if len(target.size())>2:
        if target.size(1)==3:
            y1 = get_labelmaps_with_coords(target, num_classes, on_value=on_value, off_value=off_value, device=device, label_size=label_size)
            y2 = y1.flip(0)
            # y2 = get_labelmaps_with_coords(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device, label_size=label_size)
        else:
            raise ValueError("Not supported label type")
    else:
        y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
        y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)

    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class TokenLabelMixup:
    """ Mixup/Cutmix with label that applies different params to each element or whole batch
    Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
        label_size (int): target label size
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000, label_size=1):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
        self.label_size=label_size

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, label_size=self.label_size)
        return x, target


class FastCollateTokenLabelMixup(TokenLabelMixup):
    """ Fast Collate w/ Mixup/Cutmix with label that applies different params to each element or whole batch
    Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

    A Mixup impl that's performed while collating the batches.
    """

    def _mix_elem_collate(self, output, batch, half=False):
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix[i]:
                    if not half:
                        mixed = mixed.copy()
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        if half:
            lam_batch = np.concatenate((lam_batch, np.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output, batch):
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed_i = batch[i][0]
            mixed_j = batch[j][0]
            assert 0 <= lam <= 1.0
            if lam < 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    patch_i = mixed_i[:, yl:yh, xl:xh].copy()
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    mixed_temp = mixed_i.astype(np.float32) * lam + mixed_j.astype(np.float32) * (1 - lam)
                    mixed_j = mixed_j.astype(np.float32) * lam + mixed_i.astype(np.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    np.rint(mixed_j, out=mixed_j)
                    np.rint(mixed_i, out=mixed_i)
            output[i] += torch.from_numpy(mixed_i.astype(np.uint8))
            output[j] += torch.from_numpy(mixed_j.astype(np.uint8))
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output, batch):
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.:
                if use_cutmix:
                    mixed = mixed.copy()  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed.astype(np.float32) * lam + batch[j][0].astype(np.float32) * (1 - lam)
                    np.rint(mixed, out=mixed)
            output[i] += torch.from_numpy(mixed.astype(np.uint8))
        return lam

    def __call__(self, batch, _=None):
        batch_size = len(batch)
        assert batch_size % 2 == 0, 'Batch size should be even when using this'
        half = 'half' in self.mode
        if half:
            batch_size //= 2
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
        if self.mode == 'elem' or self.mode == 'half':
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == 'pair':
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)

        if type(batch[0][1])==type(0):
            target = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        else:
            target = torch.stack([b[1] for b in batch],0)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu',label_size=self.label_size)
        target = target[:batch_size]
        return output, target
