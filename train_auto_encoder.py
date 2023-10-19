# Training DenseFuse network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from model import DeEncoder,DeDecoder
from args_fusion import args
import pytorch_msssim
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	original_imgs_path = utils.list_images(args.dataset)#路径
	train_num = 40000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)#打乱顺序
	# for i in range(5):
	i = 2
	train(i, original_imgs_path)#训练


def train(i, original_imgs_path):

	batch_size = args.batch_size#batch_size = 4

	# load network model, RGB
	in_c = 1 # channel：1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c

	AE_Encoder = DeEncoder('sum')
	AE_Decoder = DeDecoder('sum')

	print(AE_Encoder)
	print(AE_Decoder)
	optimizer1 = Adam(AE_Encoder.parameters(), args.lr)#自适应矩估计优化器
	optimizer2 = Adam(AE_Decoder.parameters(), args.lr)  # 自适应矩估计优化器

	mse_loss = torch.nn.MSELoss()#损失函数
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		AE_Encoder = AE_Encoder.cuda()
		AE_Decoder = AE_Decoder.cuda()		# densefuse_model.to(device)#模型训练使用cuda还是本机
	tbar = trange(args.epochs)#实现进度条
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		# print("image_set_ir:{}".format(type(image_set_ir)))-->list
		#开始训练
		AE_Encoder.train()
		AE_Decoder.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)#自动得到训练图像tensor类型

			count += 1
			optimizer1.zero_grad()#梯度
			optimizer2.zero_grad()  # 梯度
			img = Variable(img, requires_grad=False)
			# Variable，也就是变量，是神经网络计算图里特有的一个概念，就是Variable提供了自动求导的功能，将tensor变成variable
			# 之前如果了解过Tensorflow的读者应该清楚神经网络在做运算的时候需要先构造一个计算图谱，然后在里面进行前向传播和反向传播。
			# Variable和Tensor本质上没有区别，不过Variable会被放入一个计算图中，然后进行前向传播，反向传播，自动求导。
			# requires_grad默认fasle不对这个变量求梯度

			if args.cuda:
				img = img.cuda()
				# img=img.to(device)
			# get fusion image
			# encoder 16x16 3核
			# densefuse_model = DenseFuse_net(input_nc, output_nc)  # 加载DenseFuse模型
			en, skips = AE_Encoder(img)
			# print(len(en))
			# decoder 64x64  3核 ==> c2->c3->c4->c5
			outputs = AE_Decoder(en, skips)

			# print(outputs[0].shape)
			# resolution loss分辨率减损
			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			for output in outputs:
				#损失函数
				pixel_loss_temp = mse_loss(output, x)#像素损失函数Lp
				ssim_loss_temp = ssim_loss(output, x, normalize=True)#SSIM
				ssim_loss_value += (1-ssim_loss_temp)#Lssim
				pixel_loss_value += pixel_loss_temp
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss  L=λLssim+Lp
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
			total_loss.backward()#反向传播，得到每个参数检验的梯度
			optimizer1.step()#对其中参数进行优化
			optimizer2.step()#对其中参数进行优化


			all_ssim_loss += ssim_loss_value.item()#直接获得所对应的python数据类型
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)#进度条
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch+1 ) % (200 * args.log_interval) == 0:
				# save model
				AE_Encoder.eval()
				AE_Decoder.eval()
				AE_Encoder.cpu()
				AE_Decoder.cpu()

				save_encoder_filename = args.ssim_path[i] + '/' + "Encoder_Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_encoder_path = os.path.join(args.save_model_dir, save_encoder_filename)
				torch.save(AE_Encoder.state_dict(), save_encoder_path)

				save_decoder_filename = args.ssim_path[i] + '/' + "Decoder_Epoch_" + str(e) + "_iters_" + str(
					count) + "_" + \
										str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
											i] + ".model"
				save_decoder_path = os.path.join(args.save_model_dir, save_decoder_filename)
				torch.save(AE_Decoder.state_dict(), save_decoder_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				AE_Encoder.train()
				AE_Encoder.cuda()

				AE_Decoder.train()
				AE_Decoder.cuda()
				# densefuse_model.to(device)
				tbar.set_description("\nCheckpoint, trained model saved at", save_decoder_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})

	# save model
	AE_Encoder.eval()
	AE_Decoder.eval()
	AE_Encoder.cpu()
	AE_Decoder.cpu()

	save_encoder_filename = args.ssim_path[i] + '/' "Final_Encoder_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_encoder_path = os.path.join(args.save_model_dir, save_encoder_filename)
	torch.save(AE_Encoder.state_dict(), save_encoder_path)

	save_decoder_filename = args.ssim_path[i] + '/' "Final_Decoder_epoch_" + str(args.epochs) + "_" + \
							str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_decoder_path = os.path.join(args.save_model_dir, save_decoder_filename)
	torch.save(AE_Decoder.state_dict(), save_decoder_path)

	print("\nDone, trained model saved at", save_decoder_path)


if __name__ == "__main__":
	main()
