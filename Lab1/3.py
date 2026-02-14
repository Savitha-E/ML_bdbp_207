#Implement y = 2x12 + 3x1 + 4 and plot x1, y in the range [start=--10, stop=10, num=100]
import matplotlib.pyplot as plt
import math

start=-10
stop=10
num=100
step=((stop-start)/(num-1))

x1=[]
for i in range(num):
    x1.append(start+i*step)

print(x1)


y = []
for j in range(len(x1)):
    y.append((2 * ((x1[j]) ** 2)) + 3 * (x1[j]) + 4)

print(y)

plt.plot(x1,y)
plt.title("Plot X1 vs Y")
plt.xlabel("X1 values")
plt.ylabel("Y values")
plt.show()


