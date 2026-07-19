import numpy as np

def create_compound_image(rows, cols, limages):
    h, w, c = limages[0].shape

    compound_img = np.empty((rows * h, cols * w, c), dtype="uint8")

    for i in range(rows):
        images_max_index = min((i + 1) * cols, len(limages))
        compound_img_row = np.concatenate(limages[i * cols: images_max_index], axis=1)
        compound_img[i * h: (i + 1) * h, : compound_img_row.shape[1]] = compound_img_row

        if images_max_index == len(limages) - 1:
            break

    return compound_img

def compute_vij(H, i, j):
    v_ij = np.array([
        H[0, i]*H[i, j],
        H[0, i]*H[1, j] + H[1, i]*H[0, j],
        H[1, i]*H[1, j],
        H[2, i]*H[0, j] + H[0, i]*H[2, j],
        H[2, i]*H[1, j] + H[1, i]*H[2, j],
        H[2, i]*H[2, j]
    ])
    
    return v_ij