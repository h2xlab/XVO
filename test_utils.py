import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from params import par
from model import *
import numpy as np
from dataset import *
import glob
from tqdm import tqdm
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fisher.fisher_utils import vmf_loss as fisher_NLL
from nusc_scene_map import nusc_scene_map

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

def visualizer(ep):

    testing_data = par.test_video
    testing_data['KITTI_test'] = {'KITTI': ['03', '04', '05', '06', '07', '10']}
    
    for map, value in testing_data.items():

        for key, scenes in value.items():

            poses_dir = glob.glob('./results/{}/{}/{}/*.txt'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map))

            for dir in poses_dir:

                scene_num = dir.split('/')[-1].split('.')[0]

                if 'KITTI' in map:
                    gt_pose_path = './odom-eval/dataset/kitti/gt_poses/' + dir.split('/')[-1]
                elif 'NUSC' in map:
                    gt_pose_path = './odom-eval/dataset/nusc/gt_poses/' + dir.split('/')[-1]
                elif 'ARGO2' in map:
                    gt_pose_path = './odom-eval/dataset/argo2/gt_poses/' + dir.split('/')[-1]
                with open(gt_pose_path) as f:
                    gt_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                    
                with open(dir) as f:
                    raw_poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                        
                x_gt = gt_poses[:, 3]
                z_gt = gt_poses[:, -1]

                x_deepvo = raw_poses[:, 3]
                z_deepvo = raw_poses[:, -1]

                colors = ['b', 'y', 'r']
                descriptions = ['--', '-.', ':']

                fig, ax = plt.subplots(1, figsize=(12, 12))
                plot_route(ax,
                        x_gt, z_gt,
                        x_deepvo, z_deepvo,
                        'DeepVO',
                        scene_num,
                        colors[0],
                        descriptions[0])

                plt.savefig('./results/{}/{}/{}/{}.png'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map, scene_num))
                plt.close(fig)

def fast_test(par):

    model = VOModel()
    model = model.to('cuda')
    model_dict = model.state_dict()

    print(f"Models are loaded from {par.checkpoint_path}")
    pth_lst = sorted(glob.glob(par.checkpoint_path + '/*.pt'), reverse=True)

    for map, testing_data in par.test_video.items():
        print(map)

        df = get_test_data_info(testing_data, par.data_dir)
        dataset = TestDataset(df, (par.img_h, par.img_w))
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=par.n_processors,
            pin_memory=True,
        )
        
        for _pth in pth_lst:
            print(f'Load model {_pth}')
            ep = _pth.split('.')[0].split('-')[-1]

            if not os.path.exists('./results/{}/{}/{}'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map)):
                os.makedirs('./results/{}/{}/{}'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map))
            elif os.listdir('./results/{}/{}/{}'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map)):
                continue

            checkpoint = torch.load(_pth)
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
            model.load_state_dict(pretrained_dict)
            model.eval()

            poses_dict = {}
            with torch.no_grad():
                for step, (img_paths, x) in enumerate(tqdm(dataloader)):

                    x = x.to('cuda')
                    predicted_p, predicted_r, _, _, _, _ = model.forward(x)
                    predicted_p = predicted_p.view(-1,6).data.cpu().numpy()
                    pred_orth = fisher_NLL(predicted_r, None, overreg=1.025)
                    pred_orth = pred_orth.data.cpu().numpy()

                    for i in range(len(predicted_p)):
                        _scene = img_paths[i].split('/')[-3]
                        if _scene not in poses_dict.keys():
                            poses_dict[_scene] = []
                        
                        poses_dict[_scene].append(list(predicted_p[i]) + list(pred_orth[i].flatten()))
            
            for _scene in poses_dict.keys():
                abs_est = [[1.0,0.0,0.0,0.0,
                            0.0,1.0,0.0,0.0,
                            0.0,0.0,1.0,0.0]]
                T = np.eye(4)
                for _pose in poses_dict[_scene]:

                    R = np.array(_pose[6:]).reshape(3, 3)
                    t = np.array(_pose[:3]).reshape(3, 1)
                    T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                    T_abs = np.dot(T, T_r)
                    T = T_abs
                    abs_est.append(T[0:3, :].flatten().tolist())

                with open('./results/{}/{}/{}/{}.txt'.format(par.checkpoint_path.split('/')[-1], str(ep).zfill(3), map, _scene), 'w') as f:
                    for pose in abs_est:
                        f.write(' '.join(str(e) for e in pose))
                        f.write('\n')

            visualizer(ep)


if __name__ == '__main__':

    par.multi_modal = False
    par.checkpoint_path = "/data2/lei/DeepVO/Experiment_checkpoints/h2xlab/XVO/xvo_complete_kitti_sl_b6_lr0005"
    # par.test_video = {'ARGO2': {'ARGO2': [str(i).zfill(3) for i in range(150)]}}
    # par.test_video = {'NUSC': {'NUSC': nusc_scene_map['boston-seaport']+nusc_scene_map['singapore-queenstown']+nusc_scene_map['singapore-onenorth']}}
    # par.test_video = {'KITTI': {'KITTI': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']}}
    par.test_video = {'KITTI': {'KITTI': ['03', '04', '05', '06', '07', '10']}}
    fast_test(par)