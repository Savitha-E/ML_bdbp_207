#Implement Gaussian PDF - mean = 0, sigma = 15 in the range[start=-100, stop=100, num=100]
import matplotlib.pyplot as plt
import math

start=-100
stop=100
num=100

step=(stop-start)/(num-1)
x=[]
for i in range(num):
    x.append(start+(step*i))
print(x)

F=[]


for j in range(len(x)):
    denom = 15 * (math.sqrt(2 * math.pi))
    exponent=((x[j]-0)**2)/(2*(15**2))
    F.append((1/denom)*math.exp(-exponent))

print(F)