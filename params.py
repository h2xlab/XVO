class Parameters():
	def __init__(self):

		self.data_dir =  {'KITTI': './data/KITTI_dataset/dataset',}
		self.test_video = {'KITTI': {'KITTI': ['03', '04', '05', '06', '07', '10']}}
		self.n_processors = 16
		self.scale = 0.7
		self.img_w = 640  
		self.img_h = 384   
		self.batch_size = 6
		self.model_path = './data/XVO.pt'

par = Parameters()