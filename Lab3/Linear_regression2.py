X=[[1,1,2],[1,3,4]]
y=[[5],[11]]
learning_rate=0.1
iterations=50
#make theta column vector-----------------------------

def theta_column(X):
    theta = []
    for j in range(len(X[0])):
       theta.append([0])
    return theta

theta=theta_column(X)

#Hypothesis function------------------------------------

def hypo_fun(Xsample,theta):
    hypo_func=0
    for i in range(len(theta)):
        hypo_func+=Xsample[i]*theta[i][0]
    return hypo_func

hypo_fun_samp1=hypo_fun(X[0],theta)
hypo_fun_samp2=hypo_fun(X[1],theta)


#Cost function-------------------------------------------

def cost_fun(X,y,theta):
    m=len(X)
    summation=0
    for i in range(m):
       hypothesis_func = hypo_fun(X[i], theta)
       summation+=(hypothesis_func-y[i][0])**2

    cost=0.5*summation
    return cost
cost=cost_fun(X,y,theta)


#gradient function----------------------------------------
def grad_desc_fun(X, y, theta):
    m = len(X)        # number of samples
    n = len(theta)    # number of parameters

    # 🔹 Deep copy of theta (IMPORTANT)
    updated_theta = []
    for j in range(n):
        temp = []
        temp.append(theta[j][0])
        updated_theta.append(temp)

    # 🔹 Compute gradients
    for j in range(n):
        gradient = 0

        for i in range(m):
            h = hypo_fun(X[i], theta)   # use OLD theta
            gradient += (h - y[i][0]) * X[i][j]

        updated_theta[j][0] -= learning_rate * gradient

    return updated_theta


# ------------------ TRAINING LOOP ------------------
for iter in range(iterations):
    theta = grad_desc_fun(X, y, theta)

    if iter % 1 == 0:
        print("iteration", iter, "cost =", cost_fun(X, y, theta))


# ------------------ FINAL OUTPUT ------------------
print("\nFinal Results")
print("X =", X)
print("y =", y)
print("theta =", theta)
print("Cost =", cost_fun(X, y, theta))

print("Hypothesis sample 1 =", hypo_fun(X[0], theta))
print("Hypothesis sample 2 =", hypo_fun(X[1], theta))