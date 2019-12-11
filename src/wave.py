# coding: utf-8

import numpy as np
import tensorflow as tf

wave = np.loadtxt('../data/12_20/wave.txt')
num_examples = wave.shape[0]
model_path = '../model/20181223-122557/arc_fault-39750'
with tf.Graph().as_default(), tf.device('/gpu:1'):
	graph = tf.get_default_graph()
	graph_file_name = model_path + '.meta'
	try:
		saver = tf.train.import_meta_graph(graph_file_name)
		x = graph.get_tensor_by_name('input:0')
		tcn1_op = graph.get_tensor_by_name('TCN/tblock_0/LeakyRelu_1/Maximum:0')
		tcn2_op = graph.get_tensor_by_name('TCN/tblock_1/LeakyRelu_1/Maximum:0')
		tcn3_op = graph.get_tensor_by_name('TCN/tblock_2/LeakyRelu_1/Maximum:0')
		se_op = graph.get_tensor_by_name('SENetLayer/mul:0')
		prelogits_op = graph.get_tensor_by_name('prelogits:0')
		is_training = graph.get_tensor_by_name('training:0')

		# weights
		tcn1_conv1_weight_op = graph.get_tensor_by_name('TCN/tblock_0/conv1/kernel:0')
		tcn1_conv2_weight_op = graph.get_tensor_by_name('TCN/tblock_0/conv2/kernel:0')

		config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
		sess = tf.Session(config=config)
		saver.restore(sess, model_path)

		tcn1, tcn2, tcn3, se, prelogits, tcn1_conv1_weight, tcn1_conv2_weight = sess.run(
			[tcn1_op, tcn2_op, tcn3_op, se_op, prelogits_op, tcn1_conv1_weight_op, tcn1_conv2_weight_op],
			feed_dict={x:wave.reshape(-1, 96, 1), is_training:False})

		for i in range(num_examples):
			file_writer = open('result_{}.txt'.format(i), 'w')
			file_writer.write('tcn1\n')
			np.savetxt(file_writer, np.transpose(tcn1[i]), fmt='%.6f')
			file_writer.write('tcn2\n')
			np.savetxt(file_writer, np.transpose(tcn2[i]), fmt='%.6f')
			file_writer.write('tcn3\n')
			np.savetxt(file_writer, np.transpose(tcn3[i]), fmt='%.6f')
			file_writer.write('se\n')
			np.savetxt(file_writer, np.transpose(se[i]), fmt='%.6f')
			file_writer.write('prelogits\n')
			np.savetxt(file_writer, np.transpose(prelogits[i]), fmt='%.6f')
			file_writer.close()


		file_writer1 = open('tcn1_conv1_weight.txt', 'w')
		file_writer2 = open('tcn1_conv2_weight.txt', 'w')
		for i in range(25):
			np.savetxt(file_writer1, np.transpose(tcn1_conv1_weight[:,:,i]), fmt='%.6f')
			np.savetxt(file_writer2, np.transpose(tcn1_conv2_weight[:,:,i]), fmt='%.6f')
		file_writer1.close()
		file_writer2.close()
		
	except Exception as e:
		raise Exception(e)
	finally:
		sess.close()