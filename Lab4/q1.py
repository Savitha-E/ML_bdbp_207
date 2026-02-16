#Implement gradient descent algorithm from scratch using Python
X=[[1,1,2],[1,3,4]]
y=[[5],[11]]
learning_rate=0.1
iterations=50

#hypothesis function
#1.First create theta column. There are three features (x0,x1,x2) so three theta values to be given.
#Make a general hypothesis function so that for each sample it will return a hypothesis function for that particular sample.

def theta_column(X):
    theta=[]
    for i in range(len(X[0])):
        theta.append([0])
    return theta
theta=theta_column(X)
print(theta)

def hypothesis_gen(sample,theta): #Here we take each sample(row) and pass it. Due to this the dim of samples and theta
    hypothesis_func=0              #would match.
    for i in range(len(theta)):
       hypothesis_func+=theta[i][0]*sample[i]  #theta[row][column]It has to go to the next row because there is only one
    return hypothesis_func                  #value in each row , sample[iterate next to next within the row]
hypothesis_sample1=hypothesis_gen(X[0],theta)
hypothesis_sample2=hypothesis_gen(X[1],theta)
print(hypothesis_sample1)
print(hypothesis_sample2)

#Cost function
#write a cost function which is broken down into parts.

def cost_function(theta,X,y):
    summation=0
    for i in range(len(X)):
        hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta) #this assigns each sample (X^i) hypothesis function
        summation+=(hypothesis_func_for_all_samp-y[i][0])**2
    cost_func=0.5*summation
    return cost_func
cost_func=cost_function(theta,X,y)
print("This is my cost func",cost_func)

#Gradient function
#first make a copy of theta values into a column vector called updated theta values.
#general gradient function that updates all three theta values.
#first innerloop- (hypothesisfunction(i)-y(i))Xj^i)
#outerloop-theta
def make_updated_theta(theta):
    updated_theta = []
    for i in range(len(theta)):
        updated_theta.append([theta[i][0]])
    return updated_theta

updated_theta = make_updated_theta(theta)
print(updated_theta)

def gradient_function(updated_theta,theta,X,y):
    for j in range(len(theta)):
        summation1=0
        for i in range(len(X)):
            hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
            summation1+=(hypothesis_func_for_all_samp-y[i][0])*X[i][j]
        updated_theta[j][0]-=learning_rate*summation1
    return updated_theta
updated_theta_val=gradient_function(updated_theta,theta,X,y)
print("This is is updated theta values:",updated_theta_val)

#Run it across different iterations:
# for i in range(iterations):
#     theta_val=gradient_function(updated_theta,theta,X,y)
#     theta=theta_val
#     print("This is is updated theta values:",theta,"and this is the cost function for it",cost_function(theta,X,y))

#Convergence
def identify_convergence(theta, X, y, iterations):
    cost_fun_list = []

    for i in range(iterations):


        updated_theta = make_updated_theta(theta)

        # Compute gradient
        theta = gradient_function(updated_theta, theta, X, y)

        current_cost = cost_function(theta, X, y)
        cost_fun_list.append(current_cost)

        print("Updated theta:", theta,
              "Cost:", current_cost)

    return cost_fun_list


convergence=identify_convergence(theta,X,y,iterations)
print(convergence)