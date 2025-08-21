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
    r_center = np.zeros_like(D, dtype=np.float32)
    r_center[C] = D[C]

    # 4) 最近中心线分配（Voronoi on skeleton）
    #    对“非中心线”为 True 的布尔图做 EDT 并取 return_indices
    non_center = (~C)
    _, indices = distance_transform_edt(non_center, sampling=spacing, return_indices=True)

    R = np.zeros_like(D, dtype=np.float32)
    if mask.ndim == 3:
        iz, iy, ix = indices  # 最近中心线坐标
        R[mask] = r_center[iz[mask], iy[mask], ix[mask]]
    else:
        iy, ix = indices
        R[mask] = r_center[iy[mask], ix[mask]]

    # 5) 上界裁剪
    np.minimum(R, D, out=R)
    return R, D


def radius_change_on_centerline(D, centerline_prob, spacing, thresh=0.5, connectivity=26, agg='max'):
    """
    计算中心线上每个体素的 |dr/ds|，其它体素为 0。
    D               : np.ndarray float32, EDT(mask, sampling=spacing)，单位=mm
    centerline_prob : np.ndarray float32, 软中心线概率（或硬中心线0/1）
    spacing         : 3D:(sz,sy,sx) 或 2D:(sy,sx)（与 D 对应）
    """
    import numpy as np

    C = (centerline_prob >= thresh)
    V = np.zeros_like(D, dtype=np.float32)

    if D.ndim == 3:
        Z, Y, X = D.shape
        offs, wts = _make_offsets_weights_3d(connectivity, spacing)
        coords = np.array(np.where(C)).T  # [M,3] (z,y,x)
        for (z, y, x) in coords:
            r0 = D[z, y, x]
            if agg == 'max':
                best = 0.0
            else:
                ssum = 0.0
                cnt  = 0
            for (dz, dy, dx), w in zip(offs, wts):
                zn, yn, xn = z + int(dz), y + int(dy), x + int(dx)
                if (0 <= zn < Z) and (0 <= yn < Y) and (0 <= xn < X) and C[zn, yn, xn]:
                    slope = abs(r0 - D[zn, yn, xn]) / float(w)  # 1/mm
                    if agg == 'max':
                        if slope > best:
                            best = slope
                    else:
                        ssum += slope
                        cnt  += 1
            V[z, y, x] = best if agg == 'max' else (ssum / cnt if cnt > 0 else 0.0)

    elif D.ndim == 2:
        H, W = D.shape
        conn2d = connectivity if connectivity in (4, 8) else 8
        offs, wts = _make_offsets_weights_2d(conn2d, spacing)
        coords = np.array(np.where(C)).T  # [M,2] (y,x)
        for (y, x) in coords:
            r0 = D[y, x]
            if agg == 'max':
                best = 0.0
            else:
                ssum = 0.0
                cnt  = 0
            for (dy, dx), w in zip(offs, wts):
                yn, xn = y + int(dy), x + int(dx)
                if (0 <= yn < H) and (0 <= xn < W) and C[yn, xn]:
                    slope = abs(r0 - D[yn, xn]) / float(w)  # 1/mm
                    if agg == 'max':
                        if slope > best:
                            best = slope
                    else:
                        ssum += slope
                        cnt  += 1
            V[y, x] = best if agg == 'max' else (ssum / cnt if cnt > 0 else 0.0)

    else:
        raise ValueError("Only 2D/3D supported.")

    return V


# =========================
# Main
# =========================
if __name__ == "__main__":

    # ====== Set device ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== Step 1: Load input segmentation ======
    input_nii = "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta.nii.gz"
    tensor, affine, header = load_nii_to_tensor(input_nii)
    spacing = header.get_zooms()  # 3D: (sz, sy, sx)

    tensor_1cls = (tensor > 0).type_as(tensor)
    tensor = tensor.to(device)        # GPU（仅骨架这步用）
    tensor_1cls = tensor_1cls.to(device)

    # ====== Step 2: Extract skeleton (保持不变，用软骨架) ======
    model = SoftSkeletonize(num_iter=40).to(device)
    skeleton = model(tensor_1cls)     # (1,1,D,H,W), 0~1 概率

    # ====== Step 3: Save skeleton as NIfTI ======
    out_skel = "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta_skeleton_soft.nii.gz"
    save_tensor_to_nii(skeleton.cpu(), affine, header, out_skel)
    print(f"Done: Skeleton saved -> {out_skel}")

    # ====== Step 4: Radius map（启用 return_indices 版本；保留 Dijkstra 可切换） ======
    # 转 numpy（在 CPU 做，省显存）
    mask_np = (tensor_1cls[0,0].detach().cpu().numpy() > 0)
    soft_np = skeleton[0,0].detach().cpu().numpy().astype(np.float32)

    USE_VORONOI_INDICES = True  # ← True: 快速省显存方案；False: 用你原来的 Dijkstra
    if USE_VORONOI_INDICES:
        R_np, D_np = radius_map_voronoi_indices(mask_np, soft_np, spacing=spacing, thresh=0.5)
    else:
        R_np, D_np = radius_map_dijkstra(mask_np, soft_np, spacing=spacing, thresh=0.5, connectivity=26)

    # ====== Step 4.5: 计算中心线的 |dr/ds|（单位 1/mm）======
    V_np = radius_change_on_centerline(D_np, soft_np, spacing=spacing, thresh=0.5, connectivity=26, agg='max')

    # ====== Step 5: Save ======
    R_t = torch.from_numpy(R_np)[None, None]  # (1,1,D,H,W)
    out_radius = "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta_radius_mm_indices.nii.gz" \
                 if USE_VORONOI_INDICES else \
                 "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta_radius_mm_dijkstra.nii.gz"
    save_tensor_to_nii(R_t, affine, header, out_radius)
    print(f"Done: Radius saved -> {out_radius}")

    V_t = torch.from_numpy(V_np)[None, None]
    out_var = "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta_radius_change_abs_on_centerline_per_mm_indices.nii.gz" \
              if USE_VORONOI_INDICES else \
              "/mnt/workspace/aneurysm/nnUNetCAS/nnunetv2/tools/2016_CT1376811_cta_radius_change_abs_on_centerline_per_mm_dijkstra.nii.gz"
    save_tensor_to_nii(V_t, affine, header, out_var)
    print(f"Done: |dr/ds| saved -> {out_var}")
