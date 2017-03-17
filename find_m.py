# declare objective function

# train it

# output -> matrix to CSV file for reading in matlab

import tensorflow as tf
import numpy as np
import csv

eta = .01
gamma = .5
k = 10 # number of features
tot = 21
num_ex = 195

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def data():
	#give it all the data it needs for one batch iteration
	priv_data = []
	normal_data = []
	with open('data/priv_data_park.csv','r') as f:
		r = csv.reader(f)
		for i,row in enumerate(r):
			if i == 0:
				continue
			ex = row[k+2:]
			ex = [float(i) for i in ex]
			ex = normalize(ex)
			priv_data.append(ex)
	f.close()
	with open('data/normal_data_park.csv','r') as f:
		r = csv.reader(f)
		for i,row in enumerate(r):
			if i == 0:
				continue
			ex = row[2:k+2]
			ex = [float(i) for i in ex]
			ex = normalize(ex)
			normal_data.append(ex)		
	f.close()
	return priv_data,normal_data

p,n = data()
#batch it up
size_batch = 15
num_epochs = 100
p = np.split(np.array(p),num_ex/size_batch)
n = np.split(np.array(n),num_ex/size_batch)

#function which creates parameter objects
def model_variable(shape, name):
	variable = tf.get_variable(name=name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=tf.random_normal_initializer(stddev=.0001)
    )
	tf.add_to_collection('model_variables', variable)
	return variable

#function which calculates y_hat
def calc_obj(pi,ni):
	cost = 0
	
	priv_info = pi
	normal_data = ni
	num_features = k
	M = model_variable([num_features,num_features], 'M')

	for i in range(size_batch): # scroll through all the data
		cpi = tf.reshape(priv_info[i,:],shape=[tot-k,-1]) #current priv i
		cni = tf.reshape(normal_data[i,:],shape=[k,-1]) #current norm i
		for j in range(size_batch):
			if i >= j:
				continue
			cpj = tf.reshape(priv_info[j,:],shape=[tot-k,-1]) #current priv j
			cnj = tf.reshape(normal_data[j,:],shape=[k,-1]) #current norm j
			tmp1 = tf.subtract(tf.matmul(cpi,cpj,transpose_a=True),tf.matmul(tf.matmul(M,cni),tf.matmul(M,cnj),transpose_a=True))
			tmp1 = gamma * tf.square(tmp1)
			tmp2 = tf.subtract(tf.matmul(tf.matmul(M,cni),tf.matmul(M,cnj),transpose_a=True),tf.matmul(cni,cnj,transpose_a=True))
			tmp2 = (1-gamma) * tf.square(tmp2)
			cost = cost + tmp1 + tmp2
	return cost

priv = tf.placeholder(tf.float32, shape=[size_batch,tot-k])
norm = tf.placeholder(tf.float32, shape=[size_batch,k])
#Objective Function
loss = calc_obj(priv,norm)
model_variables = tf.get_collection('model_variables')
#Use gradient descent
optimizer = tf.train.AdamOptimizer(eta)
#Train it with respect to the model variables
train = optimizer.minimize(loss, var_list=model_variables)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


#Find the solution
for i in range(num_epochs):
	for j in range(int(num_ex/size_batch)):
		priv_batch = p[j]
		norm_batch = n[j]
		l , _ = sess.run([loss, train], feed_dict={priv : priv_batch, norm: norm_batch})

optimal_vars = sess.run(tf.get_collection('model_variables'))
print(optimal_vars)