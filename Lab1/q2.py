#Implement y = 2x1 + 3 and plot x1, y [start=-100, stop=100, num=100]

import matplotlib.pyplot as plt

start=-100
stop=100
num=100

step=int((stop-start)/(num-1))

x=[]
for i in range(start,stop,step):
    x.append(i)

print(x)

y=[]
for j in range(len(x)):
    y.append(2*x[j]+3)

print(y)


plt.plot(x,y)
plt.xlabel('x1 values')
plt.ylabel('Y values')
plt.title('Plot of x1 vs Y')
plt.show()