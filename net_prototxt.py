import sys
sys.path.append('/home/zlp/dev/caffe/python')
import os
import os.path as osp
import caffe
from caffe import layers as L, params as P, to_proto

# enum Engine {DEFAULT = 0;CAFFE = 1;CUDNN = 2;}
# enum PoolMethod { MAX = 0; AVE = 1; STOCHASTIC = 2;}
# enum Phase {TRAIN = 0;TEST = 1;}
# net.u0d_bn = L.BatchNorm(net.u0d_conv,
# 		use_global_stats=0,
# 		# moving_average_fraction=0.999, eps=1e-5,
# 		param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
def Net3DBN_S(input_dims, class_nums, ignore_label, phase="TRAIN"):
	net = caffe.NetSpec()

	############ d0 ############
	### a ###
	# net.data, net.label = L.HDF5Data(batch_size=batch_size, source=input_file, top=2, include=dict(phase=caffe.TRAIN))
	net.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	if phase == "TRAIN":
		input_dims = [input_dims[0], 1, 1, 1, 1]
		net.label = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	### b ###
	net.d0b_conv = L.Convolution(net.data,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=[1,3,3], kernel_size=[3,7,7], stride=2,
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=1)
	net.d0b_scale = L.Scale(net.d0b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0b_relu = L.ReLU(net.d0b_scale, in_place=True, engine=1)
	# # ### c ###
	# net.d0c_conv = L.Convolution(net.d0b_scale,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=64, pad=1, kernel_size=3, stride=1, 
	# 	weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	engine=1)
	# if phase == "TRAIN":
	# 	net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	# else:
	# 	net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=1)
	# net.d0c_scale = L.Scale(net.d0c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	# net.d0c_relu = L.ReLU(net.d0c_scale, in_place=True, engine=1)

	
	############ d1 ############
	### a ### First pooling
	net.d1a_pool = L.PoolingND(net.d0b_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d1b_conv = L.Convolution(net.d1a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=1)
	net.d1b_scale = L.Scale(net.d1b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1b_relu = L.ReLU(net.d1b_scale, in_place=True, engine=1)
	### c ###
	net.d1c_conv = L.Convolution(net.d1b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=1)
	net.d1c_scale = L.Scale(net.d1c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1c_relu = L.ReLU(net.d1c_scale, in_place=True, engine=1)


	############ d2 ############
	### a ###
	net.d2a_pool = L.PoolingND(net.d1c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d2b_conv = L.Convolution(net.d2a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=1)
	net.d2b_scale = L.Scale(net.d2b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2b_relu = L.ReLU(net.d2b_scale, in_place=True, engine=1)
	### c ###
	net.d2c_conv = L.Convolution(net.d2b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=1)
	net.d2c_scale = L.Scale(net.d2c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2c_relu = L.ReLU(net.d2c_scale, in_place=True, engine=1)


	############ d3 ############
	### a ### Third Pooling
	net.d3a_pool = L.PoolingND(net.d2c_scale,
		pool=1,
		kernel_size=[3,5,5],
		stride=1,
		engine=1)

	net.d3b_ip = L.InnerProduct(net.d3a_pool, num_output=class_nums)
	# net.d3c_ip = L.InnerProduct(net.d3a_pool, num_output=2)

	############ Loss ############
	if phase == "TRAIN":
		net.loss = L.SoftmaxWithLoss(net.d3b_ip, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.d3b_ip,
			phase=1)

	return net.to_proto()

def Net3DBN(input_dims, class_nums, ignore_label, phase="TRAIN"):
	net = caffe.NetSpec()

	############ d0 ############
	### a ###
	# net.data, net.label = L.HDF5Data(batch_size=batch_size, source=input_file, top=2, include=dict(phase=caffe.TRAIN))
	net.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	if phase == "TRAIN":
		input_dims = [input_dims[0], 1, 1, 1, 1]
		net.label = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	### b ###
	net.d0b_conv = L.Convolution(net.data,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=[1,3,3], kernel_size=[3,7,7], stride=2,
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=1)
	net.d0b_scale = L.Scale(net.d0b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0b_relu = L.ReLU(net.d0b_scale, in_place=True, engine=1)
	# ### c ###
	net.d0c_conv = L.Convolution(net.d0b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=1)
	net.d0c_scale = L.Scale(net.d0c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0c_relu = L.ReLU(net.d0c_scale, in_place=True, engine=1)

	
	############ d1 ############
	### a ### First pooling
	net.d1a_pool = L.PoolingND(net.d0c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d1b_conv = L.Convolution(net.d1a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=1)
	net.d1b_scale = L.Scale(net.d1b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1b_relu = L.ReLU(net.d1b_scale, in_place=True, engine=1)
	### c ###
	net.d1c_conv = L.Convolution(net.d1b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=1)
	net.d1c_scale = L.Scale(net.d1c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1c_relu = L.ReLU(net.d1c_scale, in_place=True, engine=1)


	############ d2 ############
	### a ###
	net.d2a_pool = L.PoolingND(net.d1c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d2b_conv = L.Convolution(net.d2a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=1)
	net.d2b_scale = L.Scale(net.d2b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2b_relu = L.ReLU(net.d2b_scale, in_place=True, engine=1)
	### c ###
	net.d2c_conv = L.Convolution(net.d2b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=1)
	net.d2c_scale = L.Scale(net.d2c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2c_relu = L.ReLU(net.d2c_scale, in_place=True, engine=1)


	############ d3 ############
	### a ### Third Pooling
	net.d3a_pool = L.PoolingND(net.d2c_scale,
		pool=1,
		kernel_size=[3,5,5],
		stride=1,
		engine=1)
	### b ###
	net.d3b_conv = L.Convolution(net.d3a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=512, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3b_bn = L.BatchNorm(net.d3b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3b_bn = L.BatchNorm(net.d3b_conv, use_global_stats=1)
	net.d3b_scale = L.Scale(net.d3b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3b_relu = L.ReLU(net.d3b_scale, in_place=True, engine=1)
	### c ###
	net.d3c_conv = L.Convolution(net.d3b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=512, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3c_bn = L.BatchNorm(net.d3c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3c_bn = L.BatchNorm(net.d3c_conv, use_global_stats=1)
	net.d3c_scale = L.Scale(net.d3c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3c_relu = L.ReLU(net.d3c_scale, in_place=True, engine=1)


	net.d3b_ip = L.InnerProduct(net.d3c_scale, num_output=class_nums)
	# net.d3c_ip = L.InnerProduct(net.d3a_pool, num_output=2)

	############ Loss ############
	if phase == "TRAIN":
		net.loss = L.SoftmaxWithLoss(net.d3b_ip, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.d3b_ip,
			phase=1)

	return net.to_proto()

####################################################
# batch channel z y x
input_dims = [384, 1, 24, 40, 40]
class_nums = 2
ignore_label = 255
net_name = 'Net3DBN'
net_name = 'Net3DBN_S'

dirname = '{}'.format(net_name)
prototxt_dir = osp.abspath(osp.join('Prototxt', dirname))

if not osp.exists(prototxt_dir):
	os.makedirs(prototxt_dir)
prototxt_train = osp.abspath(osp.join(prototxt_dir, 'train.prototxt'))
prototxt_test = osp.abspath(osp.join(prototxt_dir, 'test.prototxt'))

with open(prototxt_train, 'w') as f:
	if net_name == 'Net3DBN':
		net = Net3DBN(input_dims, class_nums, ignore_label, phase="TRAIN")
	elif net_name == 'Net3DBN_S':
		net = Net3DBN_S(input_dims, class_nums, ignore_label, phase="TRAIN")
	else:
		print 'net name error'
	f.write(str(net))

with open(prototxt_test, 'w') as f:
	if net_name == 'Net3DBN':
		net = Net3DBN(input_dims, class_nums, ignore_label, phase="TEST")
	elif net_name == 'Net3DBN_S':
		net = Net3DBN_S(input_dims, class_nums, ignore_label, phase="TEST")
	else:
		print 'net name error'
	f.write(str(net))