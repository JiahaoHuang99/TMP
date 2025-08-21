
import torch
import torch.nn.functional as F
import math

# ---------- 工具：生成 1D 高斯核 & 可分离 2D/3D 高斯滤波 ----------
def gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None):
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k  # (K,)

def gaussian_filter_nd(x: torch.Tensor, sigma_xyz, truncate: float = 3.0):
    """
    x: (N,C,...)  2D: (...,H,W)  3D: (...,D,H,W)
    sigma_xyz: 2D->(sy,sx)  3D->(sz,sy,sx)，单位=体素（非 mm）
    """
    dim = x.dim() - 2
    assert dim in (2, 3)
    if dim == 2:
        sy, sx = sigma_xyz
        kx = gaussian_kernel1d(sx, truncate, x.device, x.dtype)[None, None, None, :]
        ky = gaussian_kernel1d(sy, truncate, x.device, x.dtype)[None, None, :, None]
        # x-pad
        px = int(max(0, (kx.shape[-1] - 1) // 2))
        py = int(max(0, (ky.shape[-2] - 1) // 2))
        x = F.pad(x, (px, px, py, py), mode='replicate')
        x = F.conv2d(x, ky, padding=0, groups=x.shape[1])
        x = F.conv2d(x, kx, padding=0, groups=x.shape[1])
        return x
    else:
        sz, sy, sx = sigma_xyz
        kz = gaussian_kernel1d(sz, truncate, x.device, x.dtype)[None, None, :, None, None]
        ky = gaussian_kernel1d(sy, truncate, x.device, x.dtype)[None, None, None, :, None]
        kx = gaussian_kernel1d(sx, truncate, x.device, x.dtype)[None, None, None, None, :]
        pz = int(max(0, (kz.shape[2] - 1) // 2))
        py = int(max(0, (ky.shape[3] - 1) // 2))
        px = int(max(0, (kx.shape[4] - 1) // 2))
        x = F.pad(x, (px, px, py, py, pz, pz), mode='replicate')
        x = F.conv3d(x, kz, padding=0, groups=x.shape[1])
        x = F.conv3d(x, ky, padding=0, groups=x.shape[1])
        x = F.conv3d(x, kx, padding=0, groups=x.shape[1])
        return x

# ---------- 1) Soft-EDT：可导的“软距离变换”（soft-chamfer DP） ----------
@torch.no_grad()
def _build_neighbor_offsets_weights(spacing, connectivity):
    """返回邻域偏移与对应物理步长，torch 常量（CPU）。"""
    if len(spacing) == 2:
        sy, sx = spacing
        offs = []
        wts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                man = abs(dy) + abs(dx)
                if connectivity == 4 and man != 1:
                    continue
                offs.append((0, dy, dx))
                wts.append(math.sqrt((dy * sy) ** 2 + (dx * sx) ** 2))
        return offs, torch.tensor(wts, dtype=torch.float32)
    else:
        sz, sy, sx = spacing
        offs = []
        wts = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    man = abs(dz) + abs(dy) + abs(dx)
                    if connectivity == 6 and man != 1:
                        continue
                    if connectivity == 18 and man > 2:
                        continue
                    offs.append((dz, dy, dx))
                    wts.append(math.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2))
        return offs, torch.tensor(wts, dtype=torch.float32)

def softmin(stacked, tau):
    # stacked: (..., K)
    m = torch.min(stacked, dim=-1, keepdim=True).values
    z = torch.exp(-(stacked - m) / tau).sum(dim=-1)
    return m.squeeze(-1) - tau * torch.log(z + 1e-12)

def soft_chamfer_dt(mask_prob: torch.Tensor,
                    spacing,
                    iters: int = 8,
                    tau: float = None,
                    connectivity: int = 26):
    """
    可导的 EDT 近似。输入必须是软分割概率 p∈[0,1]，这样梯度能回传到 p。
    mask_prob: (B,1, H,W) 或 (B,1, D,H,W)  的 float，取值 [0,1]
    spacing:   2D=(sy,sx)；3D=(sz,sy,sx)（物理单位 mm）
    iters:     前后扫描轮数（越大越接近真 EDT，8~16 常用）
    tau:       softmin 温度（越小越接近 min；建议 tau≈min(spacing)/2）
    return:    D_soft，单位=mm；形状同 mask_prob
    """
    assert mask_prob.dim() in (4, 5) and mask_prob.size(1) == 1
    B = mask_prob.size(0)
    dim = mask_prob.dim() - 2
    if tau is None:
        tau = 0.5 * (spacing[0] if dim == 3 else min(spacing))  # 粗略默认

    # 背景“种子”概率（越接近0越像背景）
    # 这里用 1-p 作为背景权重，让 D_soft 对 p 可导
    bg = 1.0 - mask_prob.clamp(0, 1)

    # 初始化距离：背景处 ~0，其余为较大值（仍可导）
    big = torch.tensor(1e6, device=mask_prob.device, dtype=mask_prob.dtype)
    D = big.expand_as(mask_prob).clone()
    D = (1 - bg) * D  # bg≈1 → D≈0；bg≈0 → D≈big

    # 邻域步长（物理 mm）
    offs, wts_cpu = _build_neighbor_offsets_weights(spacing, connectivity)
    wts = wts_cpu.to(mask_prob.device, dtype=mask_prob.dtype)

    def sweep(D, order):
        # order: 'fwd' or 'bwd'（决定扫描方向）
        if dim == 2:
            _, _, H, W = D.shape
            ys = range(H) if order == 'fwd' else range(H - 1, -1, -1)
            xs = range(W) if order == 'fwd' else range(W - 1, -1, -1)
            for y in ys:
                for x in xs:
                    # 收集候选：自身 D[y,x] 与邻居 D[y+dy,x+dx] + w
                    cand = [D[..., 0, y, x]]
                    for (dz, dy, dx), w in zip(offs, wts):
                        yn, xn = y + dy, x + dx
                        if 0 <= yn < H and 0 <= xn < W:
                            cand.append(D[..., 0, yn, xn] + w)
                    # softmin 更新
                    stacked = torch.stack(cand, dim=-1)  # (B, K)
                    D_new = softmin(stacked, tau)        # (B,)
                    # 背景处钳为 0（可导门控）
                    gate = 1.0 - bg[..., 0, y, x]
                    D[..., 0, y, x] = gate * D_new  # bg≈1→gate≈0→保持0
        else:
            _, _, Dd, H, W = D.shape
            zs = range(Dd) if order == 'fwd' else range(Dd - 1, -1, -1)
            ys = range(H)  if order == 'fwd' else range(H - 1, -1, -1)
            xs = range(W)  if order == 'fwd' else range(W - 1, -1, -1)
            for z in zs:
                for y in ys:
                    for x in xs:
                        cand = [D[..., 0, z, y, x]]
                        for (dz, dy, dx), w in zip(offs, wts):
                            zn, yn, xn = z + dz, y + dy, x + dx
                            if 0 <= zn < Dd and 0 <= yn < H and 0 <= xn < W:
                                cand.append(D[..., 0, zn, yn, xn] + w)
                        stacked = torch.stack(cand, dim=-1)  # (B,K)
                        D_new = softmin(stacked, tau)
                        gate = 1.0 - bg[..., 0, z, y, x]
                        D[..., 0, z, y, x] = gate * D_new
        return D

    # 迭代前后扫
    for _ in range(iters):
        D = sweep(D, 'fwd')
        D = sweep(D, 'bwd')
    return D  # 单位=mm（近似 SDF 的正半部）

# ---------- 2) Soft-半径：用高斯软归属代替“最近中心线” ----------
def soft_radius_from_centerline(D_soft: torch.Tensor,
                                centerline_prob: torch.Tensor,
                                sigma_vox,
                                mask_prob: torch.Tensor = None):
    """
    D_soft:         (B,1,...)  来自 soft_chamfer_dt，单位=mm
    centerline_prob:(B,1,...)  软中心线概率 S∈[0,1]
    sigma_vox:      2D=(sy,sx) 或 3D=(sz,sy,sx)，以体素为单位的高斯半径（非 mm）
                    值越大，横截面越“扩散/平滑”，越接近 Voronoi 的硬分配
    mask_prob:      (B,1,...) 可选；若给出，用它把 R_soft 限制在前景
    return: R_soft （B,1,...），单位=mm；可导
    """
    # 中心线上的半径（= D_soft 在中心线位置）
    r_center = D_soft * centerline_prob  # (B,1,...)

    # 高斯卷积做“软最近”：num = G * (S * r), den = G * S
    num = gaussian_filter_nd(r_center, sigma_vox)
    den = gaussian_filter_nd(centerline_prob, sigma_vox)
    R_soft = num / (den + 1e-8)

    # 物理上限：R <= D_soft（此处仍可导，PyTorch 对 min 有子梯度）
    R_soft = torch.minimum(R_soft, D_soft)
    if mask_prob is not None:
        R_soft = R_soft * (mask_prob.clamp(0,1))
    return R_soft





# 已有：tensor_1cls = (B=1, C=1, D,H,W) 0/1；skeleton = (1,1,D,H,W) 软中心线；spacing=(sz,sy,sx) in mm
# 注意：soft-EDT 需要“软分割”才能反传；若你只有硬分割，可用概率 logits 的 sigmoid 作为 p

p = tensor_1cls.float()  # 如果这就是概率（0~1），直接用；否则用模型输出的概率图
D_soft = soft_chamfer_dt(p, spacing=spacing, iters=12, tau=min(spacing)/2, connectivity=26)  # (1,1,D,H,W) mm

# 软半径：sigma 用体素尺度，如 3D 数据可以设 (sz_vox, sy_vox, sx_vox) = (1.5, 1.5, 1.5)
R_soft = soft_radius_from_centerline(D_soft, skeleton, sigma_vox=(1.5,1.5,1.5), mask_prob=p)  # (1,1,D,H,W) mm

# 中心线的半径变化率：建议对 R_soft 计算 |∇R|，或复用你之前的 radius_change_on_centerline_torch，agg='mean' 更平滑可导
def grad_magnitude_nd(x, spacing):
    # x: (B,1,...)  2D/3D
    if x.dim() == 4:
        _, _, H, W = x.shape
        dy = (x[..., 1:, :] - x[..., :-1, :]) / spacing[0]
        dx = (x[..., :, 1:] - x[..., :, :-1]) / spacing[1]
        # pad回去
        dy = F.pad(dy, (0,0,0,1), mode='replicate')
        dx = F.pad(dx, (0,1,0,0), mode='replicate')
        return torch.sqrt(dx*dx + dy*dy + 1e-12)
    else:
        _, _, Dd, H, W = x.shape
        dz = (x[..., 1:, :, :] - x[..., :-1, :, :]) / spacing[0]
        dy = (x[..., :, 1:, :] - x[..., :, :-1, :]) / spacing[1]
        dx = (x[..., :, :, 1:] - x[..., :, :, :-1]) / spacing[2]
        dz = F.pad(dz, (0,0,0,0,0,1), mode='replicate')
        dy = F.pad(dy, (0,0,0,1,0,0), mode='replicate')
        dx = F.pad(dx, (0,1,0,0,0,0), mode='replicate')
        return torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-12)

# 只在中心线处取 |∇R|
abs_drds_soft = grad_magnitude_nd(R_soft, spacing=spacing) * skeleton  # (1,1,D,H,W)
