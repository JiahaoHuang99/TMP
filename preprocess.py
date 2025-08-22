import os
import nibabel as nib
import numpy as np
import scipy.ndimage
from skimage import morphology
from tqdm import tqdm
import multiprocessing as mp


def load_nii(path):
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()[:3]
    return data, spacing


def resample_to_spacing(image, spacing, target_spacing=(1.0, 1.0, 1.0), is_label=False):
    zoom_factors = [s/t for s, t in zip(spacing, target_spacing)]
    order = 0 if is_label else 1
    new_img = scipy.ndimage.zoom(image, zoom_factors, order=order)
    return new_img


def crop_or_pad(img, target_shape=(128, 256, 256)):
    in_shape = img.shape
    slices = []
    for i in range(3):
        if in_shape[i] > target_shape[i]:
            start = (in_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, in_shape[i]))
    cropped = img[slices[0], slices[1], slices[2]]

    pad_width = []
    for i in range(3):
        diff = target_shape[i] - cropped.shape[i]
        before = diff // 2
        after = diff - before
        pad_width.append((before, after))
    out = np.pad(cropped, pad_width, mode="constant", constant_values=0)
    return out


def process_case(args):
    img_path, seg_path, npz_dir, niigz_dir, target_shape, save_niigz = args
    case_id = os.path.basename(img_path).replace(".nii.gz", "")
    npz_prefix = os.path.join(npz_dir, case_id)
    nii_prefix = os.path.join(niigz_dir, case_id)

    try:
        img, spacing = load_nii(img_path)
        seg, _ = load_nii(seg_path)

        img_resampled = resample_to_spacing(img, spacing, is_label=False)
        seg_resampled = resample_to_spacing(seg, spacing, is_label=True)

        img_final = crop_or_pad(img_resampled, target_shape)
        seg_final = crop_or_pad(seg_resampled, target_shape)

        cline = morphology.skeletonize_3d(seg_final > 0).astype(np.uint8)
        cline_map = seg_final.copy()
        cline_map[cline == 0] = 0

        np.savez_compressed(
            npz_prefix + ".npz",
            image=img_final.astype(np.float32),
            seg=seg_final.astype(np.uint8),
            cline=cline.astype(np.uint8),
            cline_map=cline_map.astype(np.uint8)
        )

        if save_niigz:
            nib.save(nib.Nifti1Image(img_final.astype(np.float32), np.eye(4)),
                     nii_prefix + "_img.nii.gz")
            nib.save(nib.Nifti1Image(seg_final.astype(np.uint8), np.eye(4)),
                     nii_prefix + "_seg.nii.gz")

        return f"{case_id} done"

    except Exception as e:
        return f"{case_id} failed: {str(e)}"


def batch_process(dataset_root, npz_dir, niigz_dir,
                  target_shape=(128, 256, 256), num_workers=1, save_niigz=True):
    os.makedirs(npz_dir, exist_ok=True)
    if save_niigz:
        os.makedirs(niigz_dir, exist_ok=True)

    image_dir = os.path.join(dataset_root, "imagesTr")
    label_dir = os.path.join(dataset_root, "labelsTr")

    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii.gz")])

    tasks = []
    for img_file in img_files:
        case_id = img_file.replace(".nii.gz", "")
        img_path = os.path.join(image_dir, img_file)
        seg_path = os.path.join(label_dir, case_id + ".nii.gz")

        if not os.path.exists(seg_path):
            print(f"Warning: no label found for {case_id}, skipped.")
            continue

        tasks.append((img_path, seg_path, npz_dir, niigz_dir, target_shape, save_niigz))

    if num_workers > 1:
        print(f"Running with {num_workers} workers ...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_case, tasks), total=len(tasks)))
    else:
        print("Running in single process mode ...")
        results = [process_case(t) for t in tqdm(tasks)]

    for r in results:
        print(r)


if __name__ == "__main__":
    dataset_root = "/path/to/your/dataset_root"   # 包含 imagesTr, labelsTr
    npz_dir = "/path/to/output_npz"
    niigz_dir = "/path/to/output_niigz"

    batch_process(dataset_root, npz_dir, niigz_dir,
                  target_shape=(128, 256, 256), num_workers=8, save_niigz=True)
