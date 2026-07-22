import numpy as np
import cv2

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
        H[0, i]*H[0, j],
        H[0, i]*H[1, j] + H[1, i]*H[0, j],
        H[1, i]*H[1, j],
        H[2, i]*H[0, j] + H[0, i]*H[2, j],
        H[2, i]*H[1, j] + H[1, i]*H[2, j],
        H[2, i]*H[2, j]
    ])
    
    return v_ij


def get_corners(img_path, grid_size, criteria):
    
    im = cv2.imread(img_path)
    return_value, corners = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
    
    if not return_value:
        print(f"Pattern not found for image {img_path}")
        return None
    
    img_gray_i = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.cornerSubPix(img_gray_i,corners,(5,5),(-1,-1),criteria)
        
    return corners


def get_valid_images(images_path, grid_size):
    
    valid_images = []
    
    for p in images_path:
        
        im = cv2.imread(p)
        return_value, _ = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
        
        if return_value:
            valid_images.append(p)
        
    return valid_images
        
        
def get_cylinder(n_points, r, h, x, y, P_img):
    theta = np.linspace(0, 2*np.pi, n_points)
    
    x_b = x + r*np.cos(theta)
    x_b = x_b.reshape((n_points, 1))
    y_b = y + r*np.sin(theta)
    y_b = y_b.reshape((n_points, 1))
    z_b = np.zeros((n_points, 1))
    base_coord = np.hstack([x_b, y_b, z_b])

    x_h = x_b
    y_h = y_b
    z_h = h*np.ones((n_points, 1))
    top_coord = np.hstack([x_h, y_h, z_h])

    m_b = np.hstack([base_coord, np.ones((n_points, 1))])
    m_h = np.hstack([top_coord, np.ones((n_points, 1))])

    reprojected_cyl_b = np.zeros((n_points, 2))
    reprojected_cyl_h = np.zeros((n_points, 2))
    for i in range(n_points):
        u_proj_b = np.dot(P_img[0, :], m_b[i])/np.dot(P_img[2, :], m_b[i])
        v_proj_b = np.dot(P_img[1, :], m_b[i])/np.dot(P_img[2, :], m_b[i])
        
        u_proj_h = np.dot(P_img[0, :], m_h[i])/np.dot(P_img[2, :], m_h[i])
        v_proj_h = np.dot(P_img[1, :], m_h[i])/np.dot(P_img[2, :], m_h[i])      
        reprojected_cyl_b[i, :] = [u_proj_b, v_proj_b]
        reprojected_cyl_h[i, :] = [u_proj_h, v_proj_h]


    base_int = np.round(reprojected_cyl_b).astype(int)
    top_int = np.round(reprojected_cyl_h).astype(int)
    
    return base_int, top_int