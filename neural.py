import numpy as np


def normalization(X):
	xmax=np.max(X,axis=0)

	'''xmin=np.min(X,axis=0)
	temp=xmax-xmin
	temp[0]=1
	#print 'temp',temp
	temp=(X-xmin)/temp
	del xmin,xmax'''
	return X/xmax

def sigmoid(z):
	temp = 1+np.exp(-1*z)
	return 1/temp

def diff(z):
	return z*(1-z)

def forward(X,theta0,theta1,theta2,theta3):
	l0=X[:,np.newaxis]

	temp=np.dot(theta0,l0)


	l1=sigmoid(temp)
	l1=np.concatenate(([[1]],l1),axis=0)
	l2=sigmoid(np.dot(theta1,l1))
	l2=np.concatenate(([[1]],l2),axis=0)
	l3=sigmoid(np.dot(theta2,l2))
	l3=np.concatenate(([[1]],l3),axis=0)
	l4=sigmoid(np.dot(theta3,l3))
	
	del temp
	return l0,l1,l2,l3,l4

def calcerror(X,Y,theta0,theta1,theta2,theta3):
	error=0
	for i in range(np.shape(X)[0]):
		x=X[i,:]
		l0,l1,l2,l3,l4=forward(x,theta0,theta1,theta2,theta3)
		k=999
		for j in range(5):
			if l4[j]==np.max(l4):
				k=j+1
		if int(Y[i]) != k:
			#print i,"______________\n",y
			#print l3
			error+=1
	return error
	



X1=np.genfromtxt("features.csv",delimiter=",")
np.random.shuffle(X1)
X=X1[:,0:-1]
Y=X1[:,-1]


X=normalization(X)

X[:,0]=1
del X1

#####  CONSTANSTS   ####
m=173
#alpha
n1_hidden=926
n2_hidden=926
n3_hidden=70
np.random.seed(10)
##### Theta #####

theta0=2*np.random.random([n1_hidden,np.shape(X)[1]])-1
theta1=2*np.random.random([n2_hidden,n1_hidden+1])-1
theta2=2*np.random.random([n3_hidden,n2_hidden+1])-1
theta3=2*np.random.random([5,n3_hidden+1])-1







'''
print '\nTheta 0',theta0
print '\nTheta 1',theta1
print '\n theta 2',theta2'''
#raw_input()

X1=np.genfromtxt("test.csv",delimiter=",")
X_test=X1[:,0:-1]
Y_test=X1[:,-1]
del X1
X_test=normalization(X_test)
X_test[:,0]=1





'''
theta0=np.genfromtxt("theta0.csv",delimiter=",")
theta1=np.genfromtxt("theta1.csv",delimiter=",")
theta2=np.genfromtxt("theta2.csv",delimiter=",")
theta3=np.genfromtxt("theta3.csv",delimiter=",")
'''

temp=0
alpha=0.07
lamd=1
test_error = 100
train_error = 100



while train_error > 5:
	
	deltal3=0
	deltal2=0
	deltal1=0
	deltal0=0

	for j in range(m):
		
		x=X[j,:]		
		y=np.zeros([5,1])
		y[int(Y[j])-1]=1
		[l0,l1,l2,l3,l4]=forward(x,theta0,theta1,theta2,theta3)

		l4_error = l4-y
		l3_delta=np.dot(l4_error,np.transpose(l3))
		l3_error = (np.dot(np.transpose(theta3),l4_error)* diff(l3))[1:,:]
		
		l2_delta=np.dot(l3_error,np.transpose(l2))
		l2_error=(np.dot(np.transpose(theta2),l3_error)* diff(l2))[1:,:]
		l1_delta=np.dot(l2_error,np.transpose(l1))

		l1_error=(np.dot(np.transpose(theta1),l2_error)* diff(l1))[1:,:]
		l0_delta=np.dot(l1_error,np.transpose(l0))
		deltal3+=l3_delta
		deltal2+=l2_delta
		deltal1+=l1_delta
		deltal0+=l0_delta
		#raw_input()
	deltal3[:,1:]=(deltal3[:,1:]+lamd*theta3[:,1:])/m
	deltal3[:,0]=(deltal3[:,0])/m
	deltal2[:,1:]=(deltal2[:,1:]+lamd*theta2[:,1:])/m
	deltal2[:,0]=(deltal2[:,0])/m
	deltal1[:,1:]=(deltal1[:,1:]+lamd*theta1[:,1:])/m
	deltal1[:,0]=(deltal1[:,0])/m
	deltal0[:,1:]=(deltal0[:,1:]+lamd*theta0[:,1:])/m
	deltal0[:,0]=(deltal0[:,0])/m
	
	#print deltal1,deltal0
	theta0-=alpha*deltal0
	theta1-=alpha*deltal1
	theta2-=alpha*deltal2
	theta3-=alpha*deltal3
	temp+=1
	#raw_input("")
	train_error=calcerror(X,Y,theta0,theta1,theta2,theta3)

		


		
	

print "test error ---> " , calcerror(X_test,Y_test,theta0,theta1,theta2,theta3)

np.savetxt("theta1.csv",theta1,delimiter=",")
np.savetxt("theta0.csv",theta0,delimiter=",")
np.savetxt("theta2.csv",theta2,delimiter=",")
np.savetxt("theta3.csv",theta3,delimiter=",")


