import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from params import par
from model import *
import numpy as np
from dataset import *
import glob
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_route(ax, x_gt, z_gt, x_est, z_est, method, sequence, color, description):
    ax.scatter(x_gt[0], z_gt[0], label='Sequence start', marker='s', color='k')
    ax.plot(x_gt, z_gt, 'k', label='Ground truth', linewidth=2.5)
    ax.plot(x_est, z_est, color + '' + description, label=method, linewidth=3.5)
    ax.legend(loc='upper left', fontsize='x-large')
    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set_title('Visual Odometry - Sequence ' + sequence, fontdict={'fontsize': 20})
    ax.set_xlabel('X[m]', fontdict={'fontsize': 16})
    ax.set_ylabel('Z[m]', fontdict={'fontsize': 16})
    ax.axis('equal')

def visualizer():

    poses_dir = glob.glob('./results/KITTI/*.txt')

    for dir in poses_dir:

        scene_num = dir.split('/')[-1].split('.')[0]

        gt_pose_path = './vo-eval-tool/dataset/kitti/gt_poses/' + dir.split('/')[-1]

        with open(gt_pose_path) as f:
            gt_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
            
        with open(dir) as f:
            raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                
        x_gt = gt_poses[:, 3]
        z_gt = gt_poses[:, -1]

        x_xvo = raw_poses[:, 3]
        z_xvo = raw_poses[:, -1]

        colors = ['b', 'y', 'r']
        descriptions = ['--', '-.', ':']

        fig, ax = plt.subplots(1, figsize=(12, 12))
        plot_route(ax,
                x_gt, z_gt,
                x_xvo, z_xvo,
                'XVO',
                scene_num,
                colors[0],
                descriptions[0])

        plt.savefig('./results/KITTI/'+dir.split('/')[-1].replace('txt', 'png'))
        plt.close(fig)
            
                            
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

if __name__ == '__main__':

    model = VOModel()
    model = model.to('cuda')
    checkpoint = torch.load(par.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    model.eval()
    for map, value in par.test_video.items():

        for key, scenes in value.items():
            for s in scenes:
                df = get_test_data_info({key: [s]}, par.data_dir)
                dataset = TestDataset(df, (par.img_h, par.img_w))
                dataloader = DataLoader(
                    dataset, 
                    batch_size=par.batch_size, 
                    shuffle=False, 
                    num_workers=par.n_processors,
                    pin_memory=True,
                )

                abs_est = [[1.0,0.0,0.0,0.0,
                            0.0,1.0,0.0,0.0,
                            0.0,0.0,1.0,0.0]]

                T = np.eye(4)
                for step, x in enumerate(dataloader):
                    x = x.to('cuda')
                    predicted_pose, _ = model.forward(x)
                    predicted_pose = predicted_pose.data.cpu().numpy()

                    for i in range(len(predicted_pose)):
                        
                        t = predicted_pose[i][:3].reshape(3, 1)
                        R = eulerAnglesToRotationMatrix([predicted_pose[i][3]*3.1415926/180, predicted_pose[i][4]*3.1415926/180, predicted_pose[i][5]*3.1415926/180])
                        T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                        T_abs = np.dot(T, T_r)
                        T = T_abs
                        abs_est.append(T[0:3, :].flatten().tolist())
                        
                with open('./results/{}/{}.txt'.format(map, s), 'w') as f:
                    for pose in abs_est:
                        f.write(' '.join(str(e) for e in pose))
                        f.write('\n')
    
    visualizer()

