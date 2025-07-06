from .flownet import FlowNet
from .vodecoder import VODecoder
from .auxdecoders import UNetV0, DepthDecoder, MaskDecoder, FlowDecoder
import torch
import torch.nn as nn
import numpy as np
from fisher.fisher_utils import vmf_loss as fisher_NLL
from timm.models.vision_transformer import _create_vision_transformer
from torchvision.models.feature_extraction import create_feature_extractor
from params import par


class Distribution:
    """Interface used by different distributions"""

    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()


class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmax - vmin) * torch.rand(num_samples, device=device) + vmin
    

class VOModel(nn.Module):
    def __init__(self):
        super(VOModel, self).__init__()
        '''
        Encoder
        '''
        self.encoder = FlowNet()

        params = dict(img_size=(96, 160), patch_size=(12,16), embed_dim=256, depth=4, num_heads=4, num_classes=0, in_chans=102, class_token=False, global_pool='')
        self.transformer = _create_vision_transformer("vit_base_patch16_224_in21k", pretrained=False, **params)
        self.transformer = create_feature_extractor(self.transformer, return_nodes={"norm": "feature"}) 
        
        '''
        Decoder
        '''
        self.decoder = VODecoder(20480)
        if par.multi_modal:
            self.mask_decoder = MaskDecoder()
            self.flow_decoder = FlowDecoder()
            self.depth_decoder = DepthDecoder()

            self.audio_adapter = nn.Sequential(*[nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256*8*10, 4410*2),
                nn.ReLU(),]
                )
            self.audio_decoder = UNetV0(
                dim=1,
                in_channels=2,
                channels=[32, 32, 32],
                factors=[1, 2, 1],
                items=[1, 1, 1],
                attentions=[1, 1, 1],
                attention_heads=4,
                attention_features=64,
                resnet_groups=1,
                modulation_features=64,
                )
        
    def forward(self, x):

        batch_size = x.size(0)
        x = self.encoder(x)
        x = self.transformer(x)["feature"]
        pose, A = self.decoder(x.view(batch_size, -1))

        if par.multi_modal:
            x = x.permute(0,2,1).reshape(-1, 256, 8, 10)
            sigmas = UniformDistribution()(num_samples=1).cuda()
            audio = self.audio_decoder(self.audio_adapter(x).view(-1, 2, 4410), sigmas)
            depth = self.depth_decoder(x)
            mask = self.mask_decoder(x)
            flow = self.flow_decoder(x)

            return pose, A, audio, flow, mask, depth
        else:
            return pose, A, None, None, None, None

    def get_loss(self, x, y, audio_gt, flow_gt, mask_gt, depth_gt):

        pose, A, audio, flow, mask, depth = self.forward(x)

        losses, pred_orth = fisher_NLL(A, y[:,6:], overreg=1.025)
        loss_A = losses.mean()
        loss_p = torch.nn.functional.mse_loss(pose[:,:], y[:,:6])

        if par.multi_modal:
            #loss_m
            intersection = torch.sum(mask*mask_gt)
            dice = (2.*intersection)/(torch.sum(mask)+torch.sum(mask_gt)+1e-6)
            # print("dice", dice)
            loss_m = 1-dice

            #loss_audio
            filter = []
            fake = torch.stack((torch.from_numpy(np.array([-1.0 for _ in range(4410)])), torch.from_numpy(np.array([-1.0 for _ in range(4410)]))), 0)
            fake = fake.to('cuda')
            for i in range(len(audio_gt)):
                if torch.equal(audio_gt[i], fake):
                    filter.append(False)
                else:
                    filter.append(True)
                    
            filter_ratio = sum(filter) / len(filter)
            if filter_ratio > 0:
                audio_mse = torch.nn.functional.mse_loss(audio_gt[filter], audio[filter])
            else:
                audio_mse = torch.tensor([0.], device='cuda').float()  
            audio_mse = audio_mse * filter_ratio

            #loss_depth
            d_est, d_gt = torch.flatten(depth), torch.flatten(depth_gt)
            depth_mse = torch.nn.functional.mse_loss(d_est, d_gt)

            #loss_flow
            f_est, f_gt = torch.flatten(flow), torch.flatten(flow_gt)
            flow_loss = torch.nn.functional.mse_loss(f_est, f_gt)

            return loss_p, loss_A, audio_mse, flow_loss, loss_m, depth_mse
        else:
            return loss_p, loss_A, None, None, None, None

    def step(self, x, y, audio_gt, flow_gt, mask_gt, depth_gt, optimizer):

        optimizer.zero_grad()
        loss_p, loss_A, audio_mse, flow_loss, loss_m, depth_mse = self.get_loss(x, y, audio_gt, flow_gt, mask_gt, depth_gt)
        if par.multi_modal:
            loss = loss_p + 0.1*loss_A + 0.01*audio_mse + 0.01*flow_loss + 0.01*loss_m + 0.01*depth_mse
        else:
            loss = loss_p + 0.1*loss_A
        loss.backward()
        optimizer.step()

        return loss_p