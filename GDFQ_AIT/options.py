import os
import shutil

from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):
	def __init__(self, conf_path, args):
		super(Option, self).__init__()
		self.conf = ConfigFactory.parse_file(conf_path)
		#  ------------ General options ----------------------------------------
		self.save_path = self.conf['save_path']
		self.dataPath = self.conf['dataPath']  # path for loading data set
		self.dataset = self.conf['dataset']  # options: imagenet | cifar100
		self.nGPU = self.conf['nGPU']  # number of GPUs to use by default
		self.GPU = self.conf['GPU']  # default gpu to use, options: range(nGPU)
		self.visible_devices = args.gpu # self.conf['visible_devices']
		self.network = self.conf['network']
		# ------------- Data options -------------------------------------------
		self.nThreads = self.conf['nThreads']  # number of data loader threads
		
		# ---------- Optimization options --------------------------------------
		self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
		self.batchSize = self.conf['batchSize']  # mini-batch size
		self.momentum = self.conf['momentum']  # momentum
		if args.wd == None:
			self.weightDecay = float(self.conf['weightDecay'])  # weight decay
		else:
			self.weightDecay = float(args.wd)  # weight decay
		self.opt_type = self.conf['opt_type']
		self.warmup_epochs = self.conf['warmup_epochs']  # number of epochs for warmup

		if args.lrs == None:
			self.lr_S = self.conf['lr_S']  # initial learning rate
		else:
			self.lr_S = args.lrs  # initial learning rate

		self.lrPolicy_S = self.conf['lrPolicy_S']  # options: multi_step | linear | exp | const | step
		self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
		self.decayRate_S = self.conf['decayRate_S']  # lr decay rate
		
		# ---------- Quantization options ---------------------------------------------
		if args.qw == None:
			self.qw = self.conf['qw']
		else:
			self.qw = args.qw

		if args.qa == None:
			self.qa = self.conf['qa']
		else:
			self.qa = args.qa

		# ---------- Model options ---------------------------------------------
		self.experimentID = f"GDFQ_ait_{self.network}_qwqa_{self.qw}_{self.qa}_threshold_{args.passing_threshold}_kd_scale_{args.kd_scale}_ce_scale_{args.ce_scale}_lrS_{self.lr_S}_id_{self.conf['experimentID']}"
		self.nClasses = self.conf['nClasses']  # number of classes in the dataset
		
		# ----------KD options ---------------------------------------------
		self.temperature = self.conf['temperature']
		self.alpha = self.conf['alpha']
		self.ce_scale = args.ce_scale
		self.kd_scale = args.kd_scale
		
		# ----------Generator options ---------------------------------------------
		self.latent_dim = self.conf['latent_dim']
		self.img_size = self.conf['img_size']
		self.channels = self.conf['channels']

		self.lr_G = self.conf['lr_G']
		self.lrPolicy_G = self.conf['lrPolicy_G']  # options: multi_step | linear | exp | const | step
		self.step_G = self.conf['step_G']  # step for linear or exp learning rate policy
		self.decayRate_G = self.conf['decayRate_G']  # lr decay rate

		self.b1 = self.conf['b1']
		self.b2 = self.conf['b2']

		# -----------
		self.save = args.save
		self.passing_threshold = args.passing_threshold
		self.passing_threshold_first = args.passing_threshold
		self.threshold_decay_rate = args.threshold_decay_rate
		self.threshold_decay_ep = args.threshold_decay_ep
		self.alpha_iter = args.alpha_iter
		self.adalr = args.adalr
		
	def set_save_path(self):
		self.save_path = self.save_path + "log_{}_bs{:d}_lr{:.4f}_TELCNN_baseline_opt{}_qw{:d}_qa{:d}_epoch{}_{}/".format(
			self.dataset, self.batchSize, self.lr, self.opt_type, self.qw, self.qa,
			self.nEpochs, self.experimentID)
		
		if os.path.exists(self.save_path):
			print("{} file exist!".format(self.save_path))
			# action = input("Select Action: d (delete) / q (quit):").lower().strip()
			# act = action
			# if act == 'd':
			# 	shutil.rmtree(self.save_path)
			# else:
			# 	raise OSError("Directory {} exits!".format(self.save_path))
		
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
	
	def paramscheck(self, logger):
		logger.info("|===>The used PyTorch version is {}".format(
				self.torch_version))
		
		if self.dataset in ["cifar10", "mnist"]:
			self.nClasses = 10
		elif self.dataset == "cifar100":
			self.nClasses = 100
		elif self.dataset == "imagenet" or "thi_imgnet":
			self.nClasses = 1000
		elif self.dataset == "imagenet100":
			self.nClasses = 100