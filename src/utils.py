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

def get_valid_images(images_path, grid_size):
    
    valid_images = []
    valid_indexes = []
    
    for i, p in enumerate(images_path):
        
        im = cv2.imread(p)
        return_value, _ = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
        
        if return_value:
            valid_images.append(p)
            valid_indexes.append(i)
        
    return valid_images, valid_indexes

def get_corners(img_path, grid_size, criteria):
    
    im = cv2.imread(img_path)
    return_value, corners = cv2.findChessboardCorners(im, patternSize=grid_size, corners=None)
    
    if not return_value:
        print(f"Pattern not found for image {img_path}")
        return None
    
    img_gray_i = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.cornerSubPix(img_gray_i,corners,(5,5),(-1,-1),criteria)
        
    return corners

def compute_vij(H, i, j):
    return np.array([
        H[0, i]*H[0, j],
        H[0, i]*H[1, j] + H[1, i]*H[0, j],
        H[1, i]*H[1, j],
        H[2, i]*H[0, j] + H[0, i]*H[2, j],
        H[2, i]*H[1, j] + H[1, i]*H[2, j],
        H[2, i]*H[2, j]
    ])
    
def zhang_method(H, index_images):
    n = len(index_images)
    
    V = np.zeros((2*n, 6))

    for i, img_idx in enumerate(index_images):
        v11 = compute_vij(H[:, :, img_idx], 0, 0)
        v12 = compute_vij(H[:, :, img_idx], 0, 1)
        v22 = compute_vij(H[:, :, img_idx], 1, 1)

        V_i = np.vstack((v12, (v11-v22)))

        V[i*2:(i*2)+2, :] = V_i

    U, S, Vb = np.linalg.svd(V)

    b = Vb.transpose()[:,-1]

    if b[0] < 0:
        b = -b

    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    L = np.linalg.cholesky(B)
    L_transpose = L.T

    K = np.linalg.inv(L_transpose)

    return K / K[2, 2]

        
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

def from_R_to_Rodrigues(R):
    # axis-angle representation from robotics course
    acos_arg = np.clip((np.trace(R)-1)/2, -1.0, 1.0)
    theta = np.arccos(acos_arg)
    
    k = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2*np.sin(theta))
    
    # from the zhang's paper
    return theta*k

def from_Rodrigues_to_R(r):
    # unroll from the previous notation
    theta = np.linalg.norm(r)
    k = r/theta
    
    # skew-symmetric matrix defined by Rodrigues
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K,K)

def get_corners_stacked(corners):
    full_corners = 0
    for i in range(len(corners)):
        full_corners = np.vstack([full_corners, np.resize(corners[i], (len(corners[i])*2, 1))])  

    return full_corners[1:]

def get_real_world_coord(corners, grid_size, square_size):
    real_coords = np.zeros((len(corners[0]), 4))
    for index, corner in enumerate(corners[0]):
        u_coord = corner[0]
        v_coord = corner[1]
                    
        grid_size_cv2 = tuple(reversed(grid_size))
        u_index, v_index = np.unravel_index(index, grid_size_cv2)
                    
        x_mm = (u_index)*square_size
        y_mm = (v_index)*square_size
                    
        m_i = np.array([x_mm, y_mm, 0, 1])
        real_coords[index, :] = m_i
            
    return real_coords

def get_reprojections_stacked(real_coords, P):
    
    N = P.shape[2]
    full_projections = np.zeros((N, real_coords.shape[0], 2))
    for i in range(N):
        projections_i = np.dot(P[:, :, i], real_coords.T).T
        
        u_proj = projections_i[:, 0]/projections_i[:, 2]
        v_proj = projections_i[:, 1]/projections_i[:, 2]
        
        full_projections[i] = np.column_stack([u_proj, v_proj])

    return full_projections.reshape(-1, 1)

def stack_parameters(K, R, t):
    N = R.shape[2]
    
    K_stacked = np.array([[K[0, 0]], [K[1, 1]], [K[0, 2]], [K[1, 2]]])

    rt_stacked = 0
    for i in range(N):
        r = from_R_to_Rodrigues(R[:, :, i]).reshape(3,1)
        rt_stacked = np.vstack([rt_stacked, r, t[:, i].reshape(3,1)])
        
    rt_stacked = rt_stacked[1:]

    params = np.vstack([K_stacked, rt_stacked])
    
    return params.squeeze()

def unstack_parameters(params, N):
    alpha_u, alpha_v, u_0, v_0 = params[:4]
        
    K = np.array([[alpha_u, 0, u_0],
                    [0, alpha_v, v_0],
                    [0, 0, 1]])
        
    R = np.zeros((3, 3, N))
    t = np.zeros((3, N))
    P = np.zeros((3, 4, N))
        
    idx = 4
    for i in range(N):
        r_i = params[idx:idx+3]
        t_i = params[idx+3:idx+6]
        idx += 6
            
        R[:, :, i] = from_Rodrigues_to_R(r_i)
        t[:, i] = t_i
    
    return K, R, t
        
        
# least-square fun 
# projected_coordinates has 
def ls_fun(x, real_coords, corners):
    
    N = len(corners)
    
    K, R, t = unstack_parameters(x, N)   
    P = np.zeros((3, 4, N))
    
    for i in range(N):
        
        Q = np.dot(K, R[:, :, i])
        q = np.dot(K, t[:, i].reshape(3,1))
    
        P[:, :, i] = np.hstack([Q, q])
    
    corners_stacked = get_corners_stacked(corners)
    reprojections_stacked = get_reprojections_stacked(real_coords, P)
    
    return (corners_stacked-reprojections_stacked).squeeze()