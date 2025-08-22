        cline = morphology.skeletonize_3d(seg > 0).astype(np.uint8)
        cline_map = seg.copy()
        cline_map[cline == 0] = 0
        print(cline.shape)
        print(cline_map.shape)
        print(seg.shape)
        print(img_numpy.shape)

        # np.savez_compressed(os.path.join(out_dir, file + '.npz'), img=img_numpy, seg=seg, cline=cline_map)
