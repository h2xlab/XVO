import os
import numpy as np
import time
import math
import glob
from params import par

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z], dtype=np.float32)

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def matrix_rt(p):
    return np.vstack([np.reshape(p.astype(np.float32), (3, 4)), [[0., 0., 0., 1.]]])

def create_pose_data():
    if not os.path.exists(f'./poses'):
        os.mkdir(f'./poses')

    for key in par.data_dir.keys():
        if not os.path.exists(f'./poses/{key}'):
            os.mkdir(f'./poses/{key}')
        
        poses_dir = glob.glob(par.data_dir[key] + '/poses/*.txt')
        
        for dir in poses_dir:
            with open(dir) as f:
                raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
            
            poses = []
            for i in range(len(raw_poses)-1):
                pose1 = matrix_rt(raw_poses[i])
                pose2 = matrix_rt(raw_poses[i + 1])
                pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
                R = pose2wrt1[0:3, 0:3]
                t = pose2wrt1[0:3, 3]
                angles = rotationMatrixToEulerAngles(R)
                # poses.append(np.concatenate((t, angles))) #Radius
                poses.append(np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))) #Degree
            
            poses = np.array(poses)
            np.save(f'./poses/{key}/' + dir.split('/')[-1][:-4] +'.npy', poses)

def create_youtube_pose_data(pose_path):

    if not os.path.exists(f'./poses/YouTube'):
        os.mkdir(f'./poses/YouTube')
    
    poses_dir = glob.glob(pose_path + '/*.txt')
    
    for dir in poses_dir:
        with open(dir) as f:
            raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
        
        poses = []
        for i in range(len(raw_poses)-1):
            pose1 = matrix_rt(raw_poses[i])
            pose2 = matrix_rt(raw_poses[i + 1])
            pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
            R = pose2wrt1[0:3, 0:3]
            t = pose2wrt1[0:3, 3]
            angles = rotationMatrixToEulerAngles(R)
            # poses.append(np.concatenate((t, angles))) #Radius
            poses.append(np.concatenate((t, [180 / 3.1415926 * angles[0], 180 / 3.1415926 * angles[1], 180 / 3.1415926 * angles[2]], R.flatten()))) #Degree
        
        poses = np.array(poses)
        np.save(f'./poses/YouTube/' + dir.split('/')[-1][:-4] +'.npy', poses)

if __name__ == '__main__':

    create_pose_data()
    # create_youtube_pose_data('/home/lei/Documents/DeepVO/XVO/results/003/YouTube')