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

def hypothesis_gen(sample,theta): #Here we take each sample(row) and pass it. Due to this the dim of samples and theta
    hypothesis_func=0              #would match.
    for i in range(len(theta)):
       hypothesis_func+=theta[i][0]*sample[i]  #theta[row][column]It has to go to the next row because there is only one
    return hypothesis_func                  #value in each row , sample[iterate next to next within the row]

#Cost function
#write a cost function which is broken down into parts.

def cost_function(theta,X,y):
    summation=0
    m=len(X)
    for i in range(len(X)):
        hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta) #this assigns each sample (X^i) hypothesis function
        summation+=(hypothesis_func_for_all_samp-y[i][0])**2
    cost_func=(1/(2*m))*summation
    return cost_func

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

def gradient_function(updated_theta,theta,X,y):
    m=len(X)
    for j in range(len(theta)):
        summation1=0
        for i in range(len(X)):
            hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
            summation1+=(hypothesis_func_for_all_samp-y[i][0])*X[i][j]
        updated_theta[j][0]-=learning_rate*(1/m)*summation1
    return updated_theta

#Convergence
def identify_convergence(theta, X, y, iterations):
    for i in range(iterations):
        updated_theta = make_updated_theta(theta)
        theta = gradient_function(updated_theta, theta, X, y)
    return theta

final_theta = identify_convergence(theta,X,y,iterations)


def predict(X, theta):
    predictions=[]
    for i in range(len(X)):
        predictions.append(hypothesis_gen(X[i],theta))
    return predictions

y_pred = predict(X, final_theta)


def mean_squared_error(y_true, y_pred):
    summation=0
    m=len(y_true)
    for i in range(m):
        summation+=(y_true[i][0]-y_pred[i])**2
    return summation/m


def r2_score(y_true, y_pred):
    mean_y=0
    m=len(y_true)
    for i in range(m):
        mean_y+=y_true[i][0]
    mean_y/=m

    ss_total=0
    ss_residual=0

    for i in range(m):
        ss_total+=(y_true[i][0]-mean_y)**2
        ss_residual+=(y_true[i][0]-y_pred[i])**2

    return 1-(ss_residual/ss_total)

mse = mean_squared_error(y,y_pred)
r2 = r2_score(y,y_pred)

print("Final Theta values :",final_theta)
print("MSE:",mse)
print("R2 Score:",r2)
