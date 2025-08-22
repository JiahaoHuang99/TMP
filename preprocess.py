def np_load_any(p):
    """
    统一用 nnUNetv2 的官方IO读取 b2nd/npz/npy：
    - 返回 ndarray，形状标准化为 (C, Z, Y, X)
    """
    try:
        from nnunetv2.utilities.file_io import load_array_from_file
    except Exception as e:
        raise RuntimeError(
            "无法导入 nnunetv2 的 load_array_from_file。请先安装/激活 nnUNetv2。\n"
            "pip install nnunetv2  或确认当前环境包含该模块。"
        ) from e

    arr = load_array_from_file(p)  # 支持 .b2nd / .npy / .npz 等
    arr = np.asarray(arr)

    # 标准化到 (C, Z, Y, X)
    if arr.ndim == 3:
        # 没有通道维，补一个 C=1
        arr = arr[None, ...]                  # -> (1, Z, Y, X)
    elif arr.ndim == 4:
        # 假定是 (C, Z, Y, X)；若你的数据是 (Z, Y, X, C) 请在此处 transpose
        pass
    else:
        raise RuntimeError(f"不支持的张量形状: {arr.shape} 读取自 {p}")

    return arr
