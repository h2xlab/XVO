import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from params import par
from model import *
import numpy as np
from dataset import *
from preprocess import *
import glob
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fisher.fisher_utils import vmf_loss as fisher_NLL

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
    
    for map, value in par.data_validation.items():

        for key, scenes in value.items():

            poses_dir = glob.glob('./results/{}/{}/{}/*.txt'.format(par.model_path.split('/')[-1], str(ep).zfill(3), map))

            for dir in poses_dir:

                scene_num = dir.split('/')[-1].split('.')[0]

                if map == 'KITTI_test':
                    gt_pose_path = './vo-eval-tool/dataset/kitti/gt_poses/' + dir.split('/')[-1]
                else:
                    gt_pose_path = './vo-eval-tool/dataset/nusc/gt_poses/' + dir.split('/')[-1]

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

                plt.savefig('./results/{}/{}/{}/{}.png'.format(par.model_path.split('/')[-1], str(ep).zfill(3), map, scene_num))
                plt.close(fig)


def test(model, ep, testing=False, validation=False):

    if testing == True:
        data_dict = par.data_testing.items()
    if validation == True:
        data_dict = par.data_validation.items()
    if testing == True and validation == True:
        assert 0
    if testing == False and validation == False:
        assert 0

    model.eval()

    with torch.no_grad():

        for map, value in data_dict:

            if not os.path.exists('./results/{}/{}/{}'.format(par.model_path.split('/')[-1], str(ep).zfill(3), map)):
                os.makedirs('./results/{}/{}/{}'.format(par.model_path.split('/')[-1], str(ep).zfill(3), map))

            for key, scenes in value.items():
                for s in scenes:
                    df = get_data_info({key: [s]}, par.data_pth)
                    dataset = VisualOdometryDataset(df, (par.img_h, par.img_w), test=True)
                    dataloader = DataLoader(
                        dataset, 
                        batch_size=128, 
                        shuffle=False, 
                        num_workers=par.n_processors,
                        pin_memory=True,
                    )

                    abs_est = [[1.0,0.0,0.0,0.0,
                                0.0,1.0,0.0,0.0,
                                0.0,0.0,1.0,0.0]]
                    T = np.eye(4)
                    for step, (x, y) in enumerate(dataloader):
                        x, y = x.to('cuda'), y.to('cuda')
                        predicted_p, predicted_r = model.forward(x)
                        losses, pred_orth = fisher_NLL(predicted_r, y[:,6:], overreg=1.025)
                        predicted_p = predicted_p.data.cpu().numpy()
                        pred_orth = pred_orth.data.cpu().numpy()

                        for i in range(len(predicted_p)):
                            
                            t = predicted_p[i][:3].reshape(3, 1)
                            R = pred_orth[i]
                            # R = eulerAnglesToRotationMatrix([predicted_p[i][3]*3.1415926/180, predicted_p[i][4]*3.1415926/180, predicted_p[i][5]*3.1415926/180])
                            T_r = np.concatenate((np.concatenate([R, t], axis=1), [[0.0, 0.0, 0.0, 1.0]]), axis=0)
                            T_abs = np.dot(T, T_r)
                            T = T_abs
                            abs_est.append(T[0:3, :].flatten().tolist())

                    with open('./results/{}/{}/{}/{}.txt'.format(par.model_path.split('/')[-1], str(ep).zfill(3), map, s), 'w') as f:
                        for pose in abs_est:
                            f.write(' '.join(str(e) for e in pose))
                            f.write('\n')

if __name__ == '__main__':

    model = VOModel(par.img_h, par.img_w)
    model = model.to('cuda')

    model_path = par.model_path
    pth_lst = []
    for _video in sorted(os.listdir(model_path)):
        pth_lst = pth_lst + sorted(glob.glob(model_path + '/' + _video + '/*.pt'))

    for num, pth in enumerate(pth_lst):

        if int(pth.split('/')[-2]) > -1:
            pass
        else:
            continue

        '''
        Load model
        '''
        print('Restored model from {}:'.format(pth))

        checkpoint = torch.load(pth)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test(model, num, validation=True)
        visualizer(num)
