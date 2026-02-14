#Implement y = x1^2, plot x1, y in the range [start=--10, stop=10, num=100].
# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
# What is the value of x1 at which the function value (y) is zero.
# What do you infer from this?

import matplotlib.pyplot as plt
import math
start=-10
stop=10
num=100

step=(stop-start)/(num-1)

x1=[]

for i in range(num):
    x1.append(start+(step*i))
print(x1)

y=[]
for j in range(len(x1)):
    y.append(x1[j]**2)

print(y)

# plt.plot(x1,y)
# plt.title("Q5 Plot x1 vs Y")
# plt.xlabel("x1 values")
# plt.ylabel("Y values")

# Compute the value of derivatives at these points, x1 = -5, -3, 0, 3, 5.
a=[-5,-3,0,3,5]

tangent=[]
fa=math.pow(a,2)
fda=2*a

for k in range(len(a)):
    tangent.append(fa+fda*(a[k]-k))

print(tangent)