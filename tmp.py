import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from heapq import heappush, heappop
from scipy.ndimage import distance_transform_edt
import torch, math, heapq


# =========================
# SoftSkeletonize
# =========================
class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
    
    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for i, _ in enumerate(range(self.num_iter)):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def soft_skel_inter(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        delta_list = [skel]
        skel_list = [skel]
        img_list = [img]
        img1_list = [img1]
        for i, _ in enumerate(range(self.num_iter)):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

            delta_list.append(delta)
            skel_list.append(skel)
            img_list.append(img)
            img1_list.append(img1)

        return delta_list, skel_list, img_list, img1_list

    def forward(self, img):
        return self.soft_skel(img)

    def get_inter_mask(self, img):
        delta_list, skel_list, img_list, img1_list = self.soft_skel_inter(img)
        return delta_list, skel_list, img_list, img1_list

# =========================
# I/O Tools
# =========================
def load_nii_to_tensor(path):
    img = nib.load(path)
    data = img.get_fdata()
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # shape: (1,1,D,H,W)
    return tensor, img.affine, img.header

def save_tensor_to_nii(tensor, affine, header, out_path):
    array = tensor.squeeze().cpu().numpy()
    nib.save(nib.Nifti1Image(array, affine, header), out_path)


# =========================
# 多源 Dijkstra / FMM 半径回填（原有，保留可切换）
# =========================
def _make_offsets_weights_3d(connectivity, spacing):
    sz, sy, sx = spacing
    offs, wts = [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                manhattan = abs(dz) + abs(dy) + abs(dx)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offs.append((dz, dy, dx))
                wts.append(np.sqrt((dz*sz)**2 + (dy*sy)**2 + (dx*sx)**2))
    return np.asarray(offs, dtype=np.int8), np.asarray(wts, dtype=np.float32)

def _make_offsets_weights_2d(connectivity, spacing):
    sy, sx = spacing
    offs, wts = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            manhattan = abs(dy) + abs(dx)
            if connectivity == 4 and manhattan != 1:
                continue
            offs.append((dy, dx))
            wts.append(np.sqrt((dy*sy)**2 + (dx*sx)**2))
    return np.asarray(offs, dtype=np.int8), np.asarray(wts, dtype=np.float32)

def radius_map_dijkstra(mask, centerline_prob, spacing, thresh=0.5, connectivity=26):
    """
    在 mask 内部做“中心线多源最短路分配”，返回半径图 R（mm）和内距 D（mm）
    mask:             np.bool_ 或 {0,1}，(Z,Y,X) 或 (H,W)
    centerline_prob:  np.float32，软中心线概率（或硬中心线0/1）
    spacing:          3D:(sz,sy,sx) 2D:(sy,sx)
    """
    mask = np.asarray(mask) > 0
    prob = np.asarray(centerline_prob).astype(np.float32)

    if mask.ndim == 3:
        Z, Y, X = mask.shape
        assert len(spacing) == 3, "3D 需要 spacing=(sz,sy,sx)"
        D = distance_transform_edt(mask, sampling=spacing).astype(np.float32)  # 到边界的物理距离

        C = (prob >= thresh) & mask
        if not np.any(C):
            return D.copy(), D

        dist  = np.full(mask.shape, np.inf, dtype=np.float32)
        label = np.full(mask.shape, -1, dtype=np.int32)
        seeds = np.array(np.where(C)).T  # [M,3] (z,y,x)
        r_seed = D[C].astype(np.float32) # 每个种子的半径=该点的内切球半径

        heap = []
        for i, (z, y, x) in enumerate(seeds):
            dist[z, y, x]  = 0.0
            label[z, y, x] = i
            heappush(heap, (0.0, int(z), int(y), int(x)))

        offs, wts = _make_offsets_weights_3d(connectivity, spacing)

        while heap:
            d, z, y, x = heappop(heap)
            if d != dist[z, y, x]:
                continue
            my_label = label[z, y, x]
            for (dz, dy, dx), w in zip(offs, wts):
                zn, yn, xn = z+int(dz), y+int(dy), x+int(dx)
                if (zn < 0) or (zn >= Z) or (yn < 0) or (yn >= Y) or (xn < 0) or (xn >= X):
                    continue
                if not mask[zn, yn, xn]:
                    continue
                nd = d + float(w)
                if nd < dist[zn, yn, xn]:
                    dist[zn, yn, xn]  = nd
                    label[zn, yn, xn] = my_label
                    heappush(heap, (nd, zn, yn, xn))

        R = np.zeros_like(D, dtype=np.float32)
        in_mask = mask & (label >= 0)
        if np.any(in_mask):
            R[in_mask] = r_seed[label[in_mask]]
        miss = mask & (label < 0)
        if np.any(miss):
            R[miss] = D[miss]
        np.minimum(R, D, out=R)
        return R, D

    elif mask.ndim == 2:
        H, W = mask.shape
        assert len(spacing) == 2, "2D 需要 spacing=(sy,sx)"
        D = distance_transform_edt(mask, sampling=spacing).astype(np.float32)

        C = (prob >= thresh) & mask
        if not np.any(C):
            return D.copy(), D

        dist  = np.full(mask.shape, np.inf, dtype=np.float32)
        label = np.full(mask.shape, -1, dtype=np.int32)
        seeds = np.array(np.where(C)).T  # [M,2] (y,x)
        r_seed = D[C].astype(np.float32)

        heap = []
        for i, (y, x) in enumerate(seeds):
            dist[y, x]  = 0.0
            label[y, x] = i
            heappush(heap, (0.0, int(y), int(x)))

        offs, wts = _make_offsets_weights_2d(connectivity, spacing)

        while heap:
            d, y, x = heappop(heap)
            if d != dist[y, x]:
                continue
            my_label = label[y, x]
            for (dy, dx), w in zip(offs, wts):
                yn, xn = y+int(dy), x+int(dx)
                if (yn < 0) or (yn >= H) or (xn < 0) or (xn >= W):
                    continue
                if not mask[yn, xn]:
                    continue
                nd = d + float(w)
                if nd < dist[yn, xn]:
                    dist[yn, xn]  = nd
                    label[yn, xn] = my_label
                    heappush(heap, (nd, yn, xn))

        R = np.zeros_like(D, dtype=np.float32)
        in_mask = mask & (label >= 0)
        if np.any(in_mask):
            R[in_mask] = r_seed[label[in_mask]]
        miss = mask & (label < 0)
        if np.any(miss):
            R[miss] = D[miss]
        np.minimum(R, D, out=R)
        return R, D

    else:
        raise ValueError("Only 2D/3D supported.")


# =========================
# 新增：EDT return_indices 实现的“最近中心线分配”（更快/更省显存）
# =========================
def radius_map_voronoi_indices(mask, centerline_prob, spacing, thresh=0.5):
    """
    用 SciPy EDT 的 return_indices 做最近中心线分配：
      1) D = EDT(mask)  -> 内部到边界的物理距离（mm）
      2) C = centerline_prob>=thresh
      3) r_center = D on C
      4) 每个体素取最近的中心线坐标索引（indices），回填半径
      5) R = min(R, D)
    """
    mask = np.asarray(mask) > 0
    prob = np.asarray(centerline_prob).astype(np.float32)

    # 1) 内部 EDT（mm）
    D = distance_transform_edt(mask, sampling=spacing).astype(np.float32)

    # 2) 中心线（可来自 soft skeleton 的阈值）
    C = (prob >= thresh) & mask
    if not np.any(C):
        return D.copy(), D

    # 3) 中心线半径
    r_center = np.zeros_like(D, dt
