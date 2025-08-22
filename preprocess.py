import os, argparse, pickle, numpy as np, json

def load_pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)

def np_load_any(p):
    # 尝试直接 np.load；如果你的环境有 nnUNet 的官方 IO，可替换为其读写函数
    try:
        return np.load(p, allow_pickle=False)
    except Exception:
        return np.load(p, allow_pickle=False, mmap_mode='r')

def fg_bbox(seg):  # seg: (1,Z,Y,X) or (Z,Y,X)
    s = seg[0] if seg.ndim == 4 else seg
    idx = np.argwhere(s > 0)
    if idx.size == 0:
        return np.array([0,0,0]), np.array(s.shape)
    mn = idx.min(0); mx = idx.max(0) + 1
    return mn, mx

def expand_mm(mn, mx, spacing_zyx, shape_zyx, margin_mm=(30,30,30)):
    vox = np.round(np.array(margin_mm)/np.array(spacing_zyx)).astype(int)
    mn2 = np.maximum(mn - vox, 0)
    mx2 = np.minimum(mx + vox, np.array(shape_zyx))
    return mn2, mx2

def crop(arr, mn, mx):  # arr: (C,Z,Y,X) or (Z,Y,X)
    if arr.ndim == 4:
        return arr[:, mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    else:
        return arr[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]

def center_crop_pad(arr, target_zyx, pad_val=0):
    tz, ty, tx = target_zyx
    if arr.ndim == 4:
        C, Z, Y, X = arr.shape; out = np.full((C,tz,ty,tx), pad_val, arr.dtype)
        zs=max((tz-Z)//2,0); ys=max((ty-Y)//2,0); xs=max((tx-X)//2,0)
        ze=min(zs+Z,tz); ye=min(ys+Y,ty); xe=min(xs+X,tx)
        z0=max((Z-tz)//2,0); y0=max((Y-ty)//2,0); x0=max((X-tx)//2,0)
        out[:, zs:ze, ys:ye, xs:xe] = arr[:, z0:z0+(ze-zs), y0:y0+(ye-ys), x0:x0+(xe-xs)]
    else:
        Z, Y, X = arr.shape; out = np.full((tz,ty,tx), pad_val, arr.dtype)
        zs=max((tz-Z)//2,0); ys=max((ty-Y)//2,0); xs=max((tx-X)//2,0)
        ze=min(zs+Z,tz); ye=min(ys+Y,ty); xe=min(xs+X,tx)
        z0=max((Z-tz)//2,0); y0=max((Y-ty)//2,0); x0=max((X-tx)//2,0)
        out[zs:ze, ys:ye, xs:xe] = arr[z0:z0+(ze-zs), y0:y0+(ye-ys), x0:x0+(xe-xs)]
    return out

def save_npz(path, image, label, spacing_zyx, meta):
    np.savez_compressed(path,
        image=image.astype(np.float32),
        label=label.astype(np.uint8),
        spacing=np.asarray(spacing_zyx, np.float32),
        **meta
    )

def export_single(pre_dir, out_dir, target=(128,256,256), margin_mm=(30,30,30)):
    os.makedirs(out_dir, exist_ok=True)
    cases = [f[:-4] for f in os.listdir(pre_dir) if f.endswith(".pkl")]
    for stem in cases:
        img = np_load_any(os.path.join(pre_dir, f"{stem}.b2nd"))      # (C,Z,Y,X)
        seg = np_load_any(os.path.join(pre_dir, f"{stem}_seg.b2nd"))  # (1,Z,Y,X) or (Z,Y,X)
        props = load_pkl(os.path.join(pre_dir, f"{stem}.pkl"))
        spacing = props.get('spacing')  # 期望为 [Z,Y,X]；根据你的 props 校验

        mn, mx = fg_bbox(seg)
        shape = img.shape[1:] if img.ndim==4 else img.shape
        mn, mx = expand_mm(mn, mx, spacing, shape, margin_mm=margin_mm)

        img_roi = crop(img, mn, mx)
        seg_roi = crop(seg, mn, mx)

        img_fix = center_crop_pad(img_roi, target)
        seg_fix = center_crop_pad(seg_roi, target)

        meta = dict(case_id=stem, roi_bbox=np.asarray([*mn, *mx], np.int32))
        outp = os.path.join(out_dir, f"{stem}_128x256x256.npz")
        save_npz(outp, img_fix, seg_fix, spacing, meta)
        print("saved:", outp)

def export_multislab(pre_dir, out_dir, target=(128,256,256), margin_mm=(30,30,30), overlap_z=32):
    os.makedirs(out_dir, exist_ok=True)
    index = []
    cases = [f[:-4] for f in os.listdir(pre_dir) if f.endswith(".pkl")]
    for stem in cases:
        img = np_load_any(os.path.join(pre_dir, f"{stem}.b2nd"))
        seg = np_load_any(os.path.join(pre_dir, f"{stem}_seg.b2nd"))
        props = load_pkl(os.path.join(pre_dir, f"{stem}.pkl"))
        spacing = props.get('spacing')

        mn, mx = fg_bbox(seg)
        shape = img.shape[1:] if img.ndim==4 else img.shape
        mn, mx = expand_mm(mn, mx, spacing, shape, margin_mm=margin_mm)

        img_roi = crop(img, mn, mx)
        seg_roi = crop(seg, mn, mx)
        _, Z, Y, X = img_roi.shape if img_roi.ndim==4 else (1,)+img_roi.shape

        tz, ty, tx = target
        z0 = 0
        slab_id = 0
        while z0 < Z:
            z1 = min(z0 + tz, Z)
            # 保证最后一块也至少 tz，如果不足则回拉
            if z1 - z0 < tz and Z >= tz:
                z0 = max(Z - tz, 0)
                z1 = Z
            slicer = slice(z0, z1)

            if img_roi.ndim == 4:
                img_sub = img_roi[:, slicer, :, :]
                seg_sub = seg_roi[:, slicer, :, :]
            else:
                img_sub = img_roi[slicer, :, :]
                seg_sub = seg_roi[slicer, :, :]

            img_fix = center_crop_pad(img_sub, (tz, ty, tx))
            seg_fix = center_crop_pad(seg_sub, (tz, ty, tx))

            meta = dict(case_id=stem, slab=slab_id, z_range_in_roi=[int(z0), int(z1)], roi_bbox=np.asarray([*mn, *mx], np.int32))
            outp = os.path.join(out_dir, f"{stem}_slab{slab_id:03d}_128x256x256.npz")
            save_npz(outp, img_fix, seg_fix, spacing, meta)
            index.append(dict(file=os.path.basename(outp), case=stem, slab=slab_id, z0=int(z0), z1=int(z1)))
            print("saved:", outp)

            if z1 >= Z: break
            z0 = z1 - overlap_z
            slab_id += 1
    with open(os.path.join(out_dir, "multislab_index.json"), "w") as f:
        json.dump(index, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_dir", required=True, help="path to nnUNet_preproc/.../stage0")
    ap.add_argument("--out_dir", required=True, help="where to save npz")
    ap.add_argument("--target", type=str, default="128,256,256")
    ap.add_argument("--margin_mm", type=str, default="30,30,30")
    ap.add_argument("--mode", choices=["single","multislab"], default="single")
    ap.add_argument("--overlap_z", type=int, default=32)
    args = ap.parse_args()

    tz, ty, tx = map(int, args.target.split(","))
    mm = tuple(map(float, args.margin_mm.split(",")))

    if args.mode == "single":
        export_single(args.pre_dir, args.out_dir, target=(tz,ty,tx), margin_mm=mm)
    else:
        export_multislab(args.pre_dir, args.out_dir, target=(tz,ty,tx), margin_mm=mm, overlap_z=args.overlap_z)
