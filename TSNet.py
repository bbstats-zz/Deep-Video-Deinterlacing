import numpy as np
import tensorflow as tf
import time
import os
from scipy.misc import imsave, imread, imresize
import scipy.interpolate
from scipy.ndimage.interpolation import shift
import scipy.ndimage
import matplotlib.pyplot as plt

def parabolic(f, x):
	"""Helper function to refine a peak position in an array"""
	xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
	yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
	return (xv, yv)
def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.001)
	return tf.Variable(initial, name=name)
	
def conv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
	assert isinstance(x, tf.Tensor)
	return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)
	
def relu(x):
	assert isinstance(x, tf.Tensor)
	return tf.nn.relu(x)
	
def vupscale(x, upfield=True):
	sh = x.get_shape().as_list()
	out_size = [-1] +[sh[1]*2]+[s for s in sh[2:]]
	t = x
	if upfield:
		out = tf.concat([t, tf.zeros_like(t)],2)
	else:
		out = tf.concat([tf.zeros_like(t), t],2)
	out = tf.reshape(out,out_size)
	return out

def replaceField(x, input_image, upfield=True):

	upper_input = input_image[:,0::2,:,:]
	lower_input = input_image[:,1::2,:,:]

	if upfield:
		x = vupscale(x,upfield=False)
		upper_input = vupscale(upper_input,upfield=True)
		out = x + upper_input
	else:
		x = vupscale(x,upfield=True)
		lower_input = vupscale(lower_input,upfield=False)
		out = x + lower_input

	return out

class TSNet():

	def __init__(self):
		self.model_name = "TSNet"

	def createNet(self, input):

		self.c1 = weight_variable([3, 3, 1, 64], name='deinterlace/t_conv1_w')
		self.c2 = weight_variable([3, 3, 64, 64], name='deinterlace/t_conv2_w')
		self.c3 = weight_variable([1, 1, 64, 32], name='deinterlace/t_conv3_w')
		self.c41 = weight_variable([3, 3, 32, 32], name='deinterlace/t_conv41_w')
		self.c42 = weight_variable([3, 3, 32, 32], name='deinterlace/t_conv42_w')
		self.c51 = weight_variable([3, 3, 32, 1], name='deinterlace/t_conv51_w')
		self.c52 = weight_variable([3, 3, 32, 1], name='deinterlace/t_conv52_w')

		h = relu(conv2d(input, self.c1, name='t_conv1'))
		h = relu(conv2d(h, self.c2, name='t_conv2'))

		h = conv2d(h, self.c3, name='t_conv3')

		y = conv2d(h, self.c41, name='t_conv41')

		z = conv2d(h, self.c42, name='t_conv42')

		y = conv2d(y, self.c51, strides=[1, 2, 1, 1], name='t_conv51')
		y_full = replaceField(y, input, upfield=True)
		z = conv2d(z, self.c52, strides=[1, 2, 1, 1], name='t_conv52')
		z_full = replaceField(z, input, upfield=False)

		return (y, z, y_full, z_full)

	def finish(self):
		self.sess.close()
		
	def deinterlace_frame_pick(self, frame):
		"""
		pick a deinterlacing method according to similarity of fields
		as the neural deinterlacer does not give good results for cuts
		"""
		#just bob to get the "fields"
		out_frame1, out_frame2 = self.bob_frame(frame)
		#make greyscale and take mean along vertical axis
		#needs numpy 1.17 or so
		# (720, 576, 3) becomes (720,)
		m1 = np.mean(out_frame1, axis=(0,2) )/255
		m2 = np.mean(out_frame2, axis=(0,2) )/255
		THRESHOLD = 0.03
		#testing gave 2 peaks at .11 & .31
		#rest below 0.003
		if abs(np.mean(m1-m2)) <= THRESHOLD:
			return self.deinterlace_frame(frame)
			# return out_frame1, out_frame2
		else:
			print("bobbed frame\n")
			return out_frame1, out_frame2
	
	def deinterlace_frame(self, frame):
		"""
		takes an RGB24 unit8 array,
		returns two RRGB24 uint8 arrays
		"""
		# print("processing frame")
		s_time = time.time()
		#normalize to 1
		img = frame.astype('float32') / 255.0

		input = np.swapaxes(np.swapaxes(img,0,2),1,2)
		input = input.reshape((3, self.img_height, self.img_width, 1))
		lower, upper = self.sess.run( [self.y,self.z], feed_dict={self.x: input} )
		# print('time: {} sec'.format(time.time() - s_time))
		lower_Field = np.swapaxes(np.swapaxes(lower,1,2),0,2)
		upper_Field = np.swapaxes(np.swapaxes(upper,1,2),0,2)
		self.im1[0::2,:,:] = img[0::2,:,:]
		self.im1[1::2,:,:] = lower_Field.reshape((int(self.img_height/2), self.img_width,3))
		self.im2[1::2,:,:] = img[1::2,:,:]
		self.im2[0::2,:,:] = upper_Field.reshape((int(self.img_height/2), self.img_width,3))
		self.im1 *= 255.0
		self.im2 *= 255.0
		
		return (np.clip(self.im1, 0, 255, out=self.im1).astype('uint8'), np.clip(self.im2, 0, 255, out=self.im2).astype('uint8') )
	
	def corr_lines(self, frame):
		"""
		takes an RGB24 unit8 array,
		returns two RRGB24 uint8 arrays
		"""

		#interpolated odd lines
		frame = frame.astype('float32') / 255.0
		corrs = []
		grey = np.mean(frame, axis=(2) )
		for i in range(1, grey.shape[0], 2):
			even1 = grey[i-1,0:]
			odd = grey[i,0:]
			# res = np.correlate(np.pad(even1, 300, mode="edge"), np.pad(odd, 300, mode="edge"), mode="same")
			# res = np.correlate(even1/np.linalg.norm(even1), odd/np.linalg.norm(odd), mode="same")
			res = np.correlate(even1, odd, mode="same")
			#interpolate to get the most accurate fit
			# i_peak = parabolic(res, np.argmax(res))[0]
			i_peak = np.argmax(res)
			result = i_peak - len(res)//2
			
			
			if i+1 < grey.shape[0]:
				even2 = grey[i+1,0:]
				res2 = np.correlate(even2, odd, mode="same")
				#interpolate to get the most accurate fit
				i_peak2 = parabolic(res2, np.argmax(res2))[0]
				# i_peak2 = np.argmax(res2)
				result2 = i_peak2 - len(even2)//2
				# print(result, result2)
				result = (result+result2)/2
			corrs.append(result)
		# print(even, odd)
		# plt.imshow(grey[:,:20])
		# plt.plot(corrs)
		# plt.show()
		# self.im1[0::2,:,:] = frame[0::2,:,:]
		# self.im2[1::2,:,:] = frame[1::2,:,:]
		
		return corrs
		
	def onclick(self, event):
		if event.ydata:
			result = int(event.ydata)
			if not (result & 1):     
				result += 1
			self.selected_line = result
			print(self.selected_line)
			
	def onpress(self, event):
		# print(event.key)
		ks = ( ("n", -.1), ("m", .1) )
		event_key = event.key.lower()
		for key, val in ks:
			if event_key == key:
				i = (self.selected_line-1)//2
				self.c[i]+= val
				self.out_frame[self.selected_line,:,:] = shift(self.frame[self.selected_line,:,:], (self.c[i], 0), mode="nearest")
				
				a, b = self.bob_frame(self.out_frame)
				if self.mode == "a":
				
					self.pic.set_data(a)
				elif self.mode == "b":
					self.pic.set_data(b)
				elif self.mode == "i":
					self.pic.set_data(self.out_frame)
				plt.draw()
		if event_key == "a":
			self.mode = event_key
			a, b = self.bob_frame(self.out_frame)
			self.pic.set_data(a)
			plt.draw()
		elif event_key == "b":
			self.mode = event_key
			a, b = self.bob_frame(self.out_frame)
			self.pic.set_data(b)
			plt.draw()
		elif event_key == "i":
			self.mode = event_key
			self.pic.set_data(self.out_frame)
			plt.draw()

	def correct_lines_interactive(self, frame, c):
		"""
		takes an RGB24 unit8 array,
		returns two RRGB24 uint8 arrays
		"""
		self.mode = "i"
		self.frame = frame
		self.odd_lines = np.arange(1, frame.shape[0], 2)
		self.out_frame = np.zeros( frame.shape, dtype=frame.dtype )
		self.out_frame[:,:,:] = frame[:,:,:]
		self.c = c
		indices = np.arange(frame.shape[1])
		for i,j in enumerate(range(1, frame.shape[0], 2)):
			self.out_frame[j,:,:] = shift(frame[j,:,:], (self.c[i], 0), mode="nearest")
		self.fig = plt.figure()
		ax = self.fig.add_subplot(111)
		ax.set_title('click to select odd line, press n & m to move left & right')
		self.pic = ax.imshow(self.out_frame)
		cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		cid = self.fig.canvas.mpl_connect('key_press_event', self.onpress)
		plt.show()
		print(list(self.c))
		return self.out_frame
		
	def correct_lines(self, frame, line_offsets):
		"""
		takes and returns an image array
		also needs an integer array of line offsets
		"""
		
		out_frame = np.zeros( frame.shape, dtype=frame.dtype )
		out_frame[:,:,:] = frame[:,:,:]
		for i, j in enumerate(range(1, frame.shape[0], 2)):
			out_frame[j,:,:] = shift(frame[j,:,:], (line_offsets[i], 0), mode="nearest")
		return out_frame
		
	def bob_frame(self, frame):
		"""
		takes an RGB24 unit8 array,
		returns two RRGB24 uint8 arrays
		"""		
		#interpolated odd lines
		self.im1[0::2,:,:] = frame[0::2,:,:]
		self.im1[1:-2:2,:,:] = frame[0:-2:2,:,:]//2+frame[2::2,:,:]//2
		self.im2[1::2,:,:] = frame[1::2,:,:]
		self.im2[2:-1:2,:,:] = frame[1:-2:2,:,:]//2+frame[3::2,:,:]//2
		return (self.im1.astype('uint8'), self.im2.astype('uint8') )
		
	def set_dimensions(self, img_width, img_height, gpu=0, model="./models/TSNet_advanced.model"):
		self.img_width = img_width
		self.img_height = img_height
		# print()
		shape = (self.img_height, self.img_width, 3)
		self.im1 = np.zeros(shape, dtype = 'float32')
		self.im2 = np.zeros(shape, dtype = 'float32')
		if gpu > -1:
			device_ = '/gpu:{}'.format(gpu)
			print(device_)
		else:
			device_ = '/cpu:0'
		with tf.device(device_):
			self.x = tf.placeholder(tf.float32, shape=[3, self.img_height, self.img_width, 1])
			self.y,self.z,y_full, z_full = self.createNet(self.x)

		config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
		# config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.4
		self.sess = tf.Session(config=config)
		# Restore variables from disk.
		saver = tf.train.Saver()
		saver.restore(self.sess, model)
		print("Model restored.")
		
	def deinterlace(self, img_paths, gpu=0, model="./models/TSNet_advanced.model"):
		#set up placeholders
		img = imread(img_paths[0], mode='RGB')
		img_height, img_width, img_nchannels = img.shape
		self.set_dimensions(img_width, img_height)
		
		#assume same shape throughout
		for img_path in img_paths:
			print("processing",img_path)
			frame_0, frame_1 = self.deinterlace_frame( imread(img_path, mode='RGB') )
			input_filename, input_ext = os.path.splitext(os.path.basename(img_path))
			imsave("results/"+input_filename+"_0" + input_ext, frame_0)
			imsave("results/"+input_filename+"_1" + input_ext, frame_1)
		self.finish()
