import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn

# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d


class Generator(nn.Module):
	def __init__(self, options=None, conf_path=None):
		super(Generator, self).__init__()
		self.settings = options or Option(conf_path)
		self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		gen_input = torch.mul(self.label_emb(labels), z)
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img


class Generator_imagenet(nn.Module):
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)

		super(Generator_imagenet, self).__init__()

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img


class ExperimentDesign:
	def __init__(self, generator=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.generator = generator
		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0
		self.test_input = None

		self.unfreeze_Flag = True
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
		os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices
		
		self.settings.set_save_path()
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)
		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		self.logger.info(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
		                         batch_size=self.settings.batchSize,
		                         data_path=self.settings.dataPath,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):
		if self.settings.dataset in ["cifar100","cifar10"]:
			if self.settings.network in ["resnet20_cifar100","resnet20_cifar10"]:
				self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
				self.model = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher.eval()
			else:
				assert False, "unsupport network: " + self.settings.network

		elif self.settings.dataset in ["imagenet"]:
			if self.settings.network in ["resnet18","resnet50","mobilenetv2_w1"]:
				self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
				self.model = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
				self.model_teacher.eval()
			else:
				assert False, "unsupport network: " + self.settings.network

		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		test_error, test_loss, test5_error = self.trainer.test_teacher(0)

		try:
			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch
				self.start_epoch = 0

				if epoch < 4:
					print ("\n self.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)

				train_error, train_loss, train5_error = self.trainer.train(epoch=epoch)
				self.freeze_model(self.model)

				if self.settings.dataset in ["cifar100", "cifar10"]:
					test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				elif self.settings.dataset in ["imagenet"]:
					if epoch > self.settings.warmup_epochs - 2:
						test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
					else:
						test_error = 100
						test5_error = 100
				else:
					assert False, "invalid data set"


				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					best_ep = epoch
				
				self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
				                                                                                       100 - best_top5))

				if self.settings.save:
					print('Saving generator weight...')
					torch.save(self.trainer.generator.state_dict(),f"generator-gdfq-adalr-{self.settings.network}-th-{self.settings.passing_threshold_first}-lr-{self.settings.lr_S}.pt")


		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, best_top5


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--id', type=int, metavar='experiment_id',
	                    help='Experiment ID')
	parser.add_argument('--ce_scale', type=float, default=0.0)
	parser.add_argument('--kd_scale', type=float, default=1.0)
	parser.add_argument('--gpu', type=str, default="0")
	parser.add_argument('--lrs', type=float, default=None)
	parser.add_argument('--wd', type=float, default=None)
	parser.add_argument('--save', action="store_true")
	parser.add_argument('--passing_threshold', type=float, default=0.0001)
	parser.add_argument('--threshold_decay_rate', type=float, default=0.1)
	parser.add_argument('--threshold_decay_ep', type=int, default=[100,200,300]) # 100, 200, 300
	parser.add_argument('--alpha_iter', type=int, default=5)
	parser.add_argument('--adalr', action="store_true")
	parser.add_argument('--qw', type=int, default=None)
	parser.add_argument('--qa', type=int, default=None)

	args = parser.parse_args()
	print(args)
	option = Option(args.conf_path, args)
	option.manualSeed = args.id + 1
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	if option.dataset in ["cifar100", "cifar10"]:
		generator = Generator(option)
		# option.passing_threshold = 0.001
		# option.passing_threshold_first = 0.001
		# print("GI threshold changed, different from arg")
		
	elif option.dataset in ["imagenet"]:
		generator = Generator_imagenet(option)
		# option.passing_threshold = 0.0001
		# option.passing_threshold_first = 0.0001
		# print("GI threshold changed, different from arg")
	else:
		assert False, "invalid data set"

	experiment = ExperimentDesign(generator, option)
	experiment.run()


if __name__ == '__main__':
	main()
