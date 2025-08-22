import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resample_to_spacing(img, spacing=(1.5, 1.0, 1.0), is_label=False):
    """重采样到统一 spacing"""
    sitk_img = sitk.GetImageFromArray(img.get_fdata().astype(np.float32))
    sitk_img.SetSpacing(img.header.get_zooms())

    orig_spacing = sitk_img.GetSpacing()
    orig_size = sitk_img.GetSize()
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    sitk_resampled = resample.Execute(sitk_img)
    resampled_np = sitk.GetArrayFromImage(sitk_resampled)
    return resampled_np, sitk_resampled.GetSpacing()

def crop_to_nonzero(image_np, label_np, margin=10):
    """根据 label 裁剪 + margin"""
    coords = np.argwhere(label_np > 0)
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    min_z = max(min_z - margin, 0)
    min_y = max(min_y - margin, 0)
    min_x = max(min_x - margin, 0)
    max_z = min(max_z + margin, image_np.shape[0] - 1)
    max_y = min(max_y + margin, image_np.shape[1] - 1)
    max_x = min(max_x + margin, image_np.shape[2] - 1)

    image_cropped = image_np[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    label_cropped = label_np[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    return image_cropped, label_cropped

def pad_or_crop_to_shape(image_np, label_np, target_shape=(128, 256, 256)):
    """CenterCrop / Pad 到固定大小"""
    def process(arr, target_shape):
        out = np.zeros(target_shape, dtype=arr.dtype)
        in_shape = arr.shape
        offset = [(target_shape[i] - in_shape[i]) // 2 for i in range(3)]

        z_min = max(-offset[0], 0); z_max = min(in_shape[0], target_shape[0]-offset[0])
        y_min = max(-offset[1], 0); y_max = min(in_shape[1], target_shape[1]-offset[1])
        x_min = max(-offset[2], 0); x_max = min(in_shape[2], target_shape[2]-offset[2])

        out_z_min = max(offset[0], 0); out_z_max = out_z_min + (z_max - z_min)
        out_y_min = max(offset[1], 0); out_y_max = out_y_min + (y_max - y_min)
        out_x_min = max(offset[2], 0); out_x_max = out_x_min + (x_max - x_min)

        out[out_z_min:out_z_max, out_y_min:out_y_max, out_x_min:out_x_max] = arr[z_min:z_max, y_min:y_max, x_min:x_max]
        return out

    image_fixed = process(image_np, target_shape)
    label_fixed = process(label_np, target_shape)
    return image_fixed, label_fixed

def process_case(image_path, label_path, out_npz_dir, out_nii_dir, target_shape=(128,256,256)):
    case_id = os.path.basename(image_path).replace(".nii.gz", "")
    print(f"Processing {case_id}...")

    img = nib.load(image_path)
    lbl = nib.load(label_path)

    # Step1: 重采样
    image_resampled, spacing = resample_to_spacing(img, is_label=False)
    label_resampled, _ = resample_to_spacing(lbl, is_label=True)

    # Step2: ROI + margin
    image_cropped, label_cropped = crop_to_nonzero(image_resampled, label_resampled)

    # Step3: CenterCrop / Pad
    image_fixed, label_fixed = pad_or_crop_to_shape(image_cropped, label_cropped, target_shape)

    # Step4: 保存 npz
    npz_out = os.path.join(out_npz_dir, case_id + ".npz")
    np.savez_compressed(npz_out, image=image_fixed[None], label=label_fixed[None])  # 加 channel 维度

    # Step5: 保存处理后的 nii.gz 便于检查
    img_out = nib.Nifti1Image(image_fixed.astype(np.float32), np.eye(4))
    lbl_out = nib.Nifti1Image(label_fixed.astype(np.uint8), np.eye(4))
    nib.save(img_out, os.path.join(out_nii_dir, case_id + "_image.nii.gz"))
    nib.save(lbl_out, os.path.join(out_nii_dir, case_id + "_label.nii.gz"))

    print(f"Saved {case_id}: npz + nii.gz")

if __name__ == "__main__":
    input_images = "dataset_raw/imagesTr"
    input_labels = "dataset_raw/labelsTr"
    out_npz_dir = "dataset_processed/npz"
    out_nii_dir = "dataset_processed/preprocessed_nii"

    mkdir(out_npz_dir)
    mkdir(out_nii_dir)

    for fname in os.listdir(input_images):
        if fname.endswith(".nii.gz"):
            image_path = os.path.join(input_images, fname)
            label_path = os.path.join(input_labels, fname)  # 假设 image/label 文件名对应
            process_case(image_path, label_path, out_npz_dir, out_nii_dir)
