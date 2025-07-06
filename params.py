from argo2_logs_map import argo2_test_logs_map
from nusc_scene_map import nusc_scene_map

class Parameters():
	def __init__(self):

		self.data_dir =  {
			'KITTI': '/data/lei/DeepVO/KITTI_dataset/dataset',
			'NUSC': '/data/lei/DeepVO/nuScenes_dataset/NUSC_12hz/CAM_FRONT',
			'YouTube': '/data/lei/DeepVO/YouTube/YouTube_Trim_10hz/ffmpeg',
			'ARGO2': '/data/lei/DeepVO/Argoverse_2/ARGO_test'
            }
		
		self.train_video = {
			'KITTI': ['00', '02', '08', '09'],
			# 'NUSC': nusc_scene_map['singapore-hollandvillage'],
			# 'YouTube': ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'],
            }
		
		self.test_video = {'KITTI_test': {'KITTI': ['03', '04', '05', '06', '07', '10']}}

		self.n_processors = 16
		self.scale = [0.7, 1]
		self.img_w = 640  
		self.img_h = 384   

		self.multi_modal = False
		self.seed = 2023
		self.epochs = 150
		self.learning_rate = 0.0005
		self.batch_size = 6
		self.checkpoint_path = '/data2/lei/DeepVO/Experiment_checkpoints/h2xlab/XVO/xvo_complete_kitti_sl_b6_lr0005'
		self.pretrained_flownet_path = '/data/lei/DeepVO/FlowNet_checkpoints/MaskFlownet'

par = Parameters()