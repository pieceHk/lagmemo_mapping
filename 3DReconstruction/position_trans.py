import numpy as np
    
def quaternion_to_rotation_matrix(q):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return R
    
def get_first_frame(gt_pose_path):
    # gt_pose_path = "/home/sfs/SplaTAM/data/gs_data_all/0/local_pos.txt"
    c2w_0 = np.eye(4)
    R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    with open(gt_pose_path, "r") as f:
        line = f.readline()
        line = line.split()
        q = line[1:5]
        q = np.array([float(x) for x in q])
        p = line[5:]
        p = np.array([float(x) for x in p])
        R = quaternion_to_rotation_matrix(q)
        c2w_0[:3, :3] = np.matmul(R, R_RUB_to_RDF)
        c2w_0[:3, 3] = np.matmul(R_RUB_to_RDF, p)
        
    c2w_0_inv = np.linalg.inv(c2w_0)
    R_0 = c2w_0_inv[:3, :3]
    t_0 = c2w_0_inv[:3, 3]
    R_0_inv = np.linalg.inv(R_0)
    
    return R_0, t_0, R_0_inv

    
def transform_position(position, t_0, R_0_inv):
    """
        position: (3,)
    """
    R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    position = np.matmul(R_0_inv, position - t_0)
    position = np.matmul(R_RUB_to_RDF, position)
    return position

if __name__ == "__main__":

    gt_pose_path = "/home/sfs/data_for_gsmodel/room1_wxl/data_gs/local_pos.txt"
    
    t = np.array([-4.96, 0.0, 3.16])
    _, t_0, R_0_inv = get_first_frame(gt_pose_path)
    
    print(transform_position(t, t_0, R_0_inv))