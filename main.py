import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from model import *
from dataset import *
import numpy as np
import random
import time
from params import par
import torch
import torch.optim as optim
from tqdm import tqdm

'''
Reproducibility
'''
torch.manual_seed(par.seed)
np.random.seed(par.seed)
random.seed(par.seed)
# torch.backends.cudnn.benchmark = False # may reduce performance
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True) # may reduce performance

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(par.seed)


def train(model, flowmodel, depthmodel, optimizer, scheduler, train_dl, test_dl):
    
    for ep in range(par.epochs):

        model.train()
        train_p_mean_loss = 0.0
        train_bar = tqdm(train_dl)
        st_t = time.time()
        for step, (x, y, a, m) in enumerate(train_bar):

            x, y, a, m = x.to('cuda'), y.to('cuda'), a.float().to('cuda'), m.to('cuda')
            if par.multi_modal:
                with torch.no_grad():
                    flow_gt = flowmodel(x)

                    depth1_gt = depthmodel(x[:,:3,:,:])
                    depth2_gt = depthmodel(x[:,3:,:,:])
                    depth_gt = torch.cat((depth1_gt['pred_d'], depth2_gt['pred_d']), 1)

                loss_p = model.step(x, y, a, flow_gt, m, depth_gt, optimizer).data.cpu().numpy()
            else:
                loss_p = model.step(x, y, None, None, None, None, optimizer).data.cpu().numpy()
            train_p_mean_loss += float(loss_p)
            train_bar.set_postfix({"train_p_loss": train_p_mean_loss/(step+1)})

        train_p_mean_loss /= len(train_dl)

        '''
        KITTI Test
        '''
        # test(model, testing_data, ep)
        # visualizer(ep)

        '''
        Update lr
        '''
        scheduler.step()

        '''
        Save model
        '''
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_p_mean_loss,
            'scheduler': scheduler,
        }, par.checkpoint_path + '/model_ep-{}.pt'.format(str(ep).zfill(3)))

        
        print('*' * 50)
        print("Saved checkpoint for epoch {}: {}".format(str(ep).zfill(3), par.checkpoint_path))
        print('Epoch {} takes {:.1f} sec'.format(ep, time.time()-st_t))
        print('*' * 50)
    
    return 0

if __name__ == '__main__':

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(par.checkpoint_path):
        os.makedirs(par.checkpoint_path)

    '''
    Init model, optimizer, scheduler
    '''
    model = VOModel()
    model = model.to('cuda')
    model_dict = model.state_dict()
    checkpoint = torch.load(par.pretrained_flownet_path + '/8caNov12-1532_300000.pth')
    pretrained_w = {}
    for key in checkpoint.keys():
        pretrained_w['encoder.maskflownet.'+key] = checkpoint[key]  
    pretrained_dict = {k: v for k, v in pretrained_w.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    optimizer = optim.SGD(model.parameters(), lr=par.learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    if par.multi_modal:
        depthmodel = GLPDepth(max_depth=1, is_train=False).to('cuda')
        model_weight = torch.load(par.pretrained_flownet_path + '/best_model_nyu.ckpt')
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        depthmodel.load_state_dict(model_weight)

        flowmodel = FlowModel()
        flowmodel = flowmodel.to('cuda')
        flowmodel_dict = flowmodel.state_dict()
        checkpoint = torch.load(par.pretrained_flownet_path + '/8caNov12-1532_300000.pth')
        pretrained_w = {}
        for key in checkpoint.keys():
            pretrained_w['flowestimate.'+key] = checkpoint[key]  
        pretrained_dict = {k: v for k, v in pretrained_w.items() if k in flowmodel_dict.keys()}
        flowmodel_dict.update(pretrained_dict)
        flowmodel.load_state_dict(flowmodel_dict)

        flowmodel.eval()
        depthmodel.eval()
    '''
    Generate training dataloader
    '''
    train_df = get_data_info(par.train_video, par.data_dir)
    train_dataset = VisualOdometryDataset(train_df, (par.img_h, par.img_w))
    train_dl = DataLoader(
        train_dataset, 
        batch_size=par.batch_size, 
        shuffle=True, 
        num_workers=par.n_processors,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # test_df = get_data_info(par.test_video, par.data_dir)
    # test_dataset = VisualOdometryDataset(test_df, (par.img_h, par.img_w), test=True)
    # test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=par.n_processors, pin_memory=True)
    testing_data = {'KITTI_test': {'KITTI': ['03', '04', '05', '06', '07', '10']}}

    if par.multi_modal:
        train(model, flowmodel, depthmodel, optimizer, scheduler, train_dl, testing_data)
    else:
        train(model, None, None, optimizer, scheduler, train_dl, testing_data)