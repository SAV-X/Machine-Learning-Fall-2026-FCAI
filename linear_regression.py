import numpy as np
import matplotlib.pyplot as plt


#Initialize the wrong guess line
# x = np.arange(1,11)
# y = np.arange(1,11)

x = np.arange(0,20)
# y = np.random.randint(0,10,20)
y = np.array([1,8,5,7,6,17,13,20,9,11,12,6,8,11,4,6,10,20,14,15])

plt.figure()
plt.subplot(1,2,1)
plt.plot(x,y,'or',markersize = 8, label = 'Data points')

np.random.seed(2)
w = np.random.rand(1)[0]   #initial weight we start with
b = np.random.rand(1)[0]

 
eta =  0.01
epochs = 1000
costVal = np.zeros(epochs)

for update in range(epochs):
    y_hat = (w * x) + b 
    w = w - eta * ( (y_hat - y) @ x) / len(x)
    b= b - eta * (np.mean(y_hat - y)) #mean method here because in w we have to solve things manually  

    costVal[update] = 0.5 * (np.mean((y_hat - y)**2))
    plt.subplot(1,2,1)
    plt.plot(x,y_hat,label = 'Fitted line')
    
    plt.subplot(1,2,2)
    plt.title('Cost optimization')
    plt.plot(update,costVal[update],'b.')
    plt.xlabel('Updates')
    plt.pause(0.001)
    print('w: ',w,' b: ',b,' cost: ',costVal[update])
    print()

    
    if costVal[update] < 1e-5:  #satisfying error
        break

print('Optimal w:',w)
print('Optimal b:',b)
#now we suppose to have the optimal w and b:

xvals = np.arange(0,20)
yvals = (w * xvals) + b

plt.subplot(1,2,1)
plt.plot(xvals,yvals,'r-',linewidth = 2, label = 'Fitted line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear regression using Gradient descent')
plt.grid(True, alpha = 0.3)




plt.show()