
import numpy as np
import os
import Image  
import math
import matplotlib.pyplot as plt

def plot_dot(dot_num, dot, size, mark):
	#for each point figure out the number of each value
	dot_num = []
	#for each point draw the histogram
	max_value = max(dot)
	min_value = min(dot)
	interval = (max_value - min_value)/(20.0)
	for k in range(0, 20):
		start = min_value + interval*k
		end = start + interval
		num = 0
		for j in range(0, size):
			if dot[j] > start and dot[j] < end:
				num += 1
		dot_num.append(num)
		x = (start + end)/2
		if num == 0:
			plt.plot(x, num, mark)
		else:
			plt.plot(x, pow(num, 1.3), mark)
	return dot_num


def main():
	f2 = np.fromfile('./data/layer3out_faces.bin', dtype = np.float32)
	all_dot = 8*72*72
	pos_example_num = 200
	neg_example_num = 0
	layer3out = np.array(f2).reshape((pos_example_num + neg_example_num, all_dot))
	#the number of dot_set is 42*42*8, each dot contains 3400 values
	
	pos_dot_set = []	
	neg_dot_set = []
	for i in range(0, all_dot):
		pos_dot = []
		for j in range(0, pos_example_num):
			#dot.append(1.0)
			pos_dot.append(layer3out[j][i])
		pos_dot_set.append(pos_dot)
		np_pos_dot = np.asarray(pos_dot, dtype = np.float32)
#		plt.hist(np_pos_dot,40, color = 'b', histtype = 'step')
#		plt.xlabel('postive sample')
		
		neg_dot = []
		for j in range(pos_example_num, neg_example_num + pos_example_num):
			#dot.append(1.0)
			neg_dot.append(layer3out[j][i])
		neg_dot_set.append(neg_dot)
		np_neg_dot = np.asarray(neg_dot, dtype = np.float32)
#		plt.hist(np_neg_dot,40,color= 'g', histtype = 'step')
#		plt.xlabel('negtive sample')
#		plt.show()
#		plt.savefig('./picture/curve/' + str(i)+'.jpg')
#		plt.clf()

	dot_set_num = []
	for i in range(0, all_dot):
		pos_dot_num = []
		neg_dot_num = []
		plot_dot(pos_dot_num, pos_dot_set[i], pos_example_num, 'bo')
	#	plot_dot(neg_dot_num, neg_dot_set[i], neg_example_num, 'g*')
					
		#plt.show()

		#np_dot_num = np.asarray(dot_num, dtype = np.uint8)
		#print np_dot_num
		#dot_set_num.append(dot_num)
		plt.savefig('./picture/curve/' + str(i)+'.jpg')
		plt.clf()
		
	print dot_set_num 		

if __name__ == "__main__":
	main()
	
	
	
	
	
	
	
