import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

test_point_arr = [.5,2]

epsilon_variance = np.array([2,1])
epsilon_arr = []
epsilon_arr.append(np.random.normal(loc=0,scale=epsilon_variance[0], size=(1,)))
epsilon_arr.append(np.random.normal(loc=0,scale=epsilon_variance[1], size=(1,)))
epsilon_arr = np.array(epsilon_arr)
epsilon = epsilon_arr.ravel()

for test_point in test_point_arr:
    print(f"Plot for test point = {test_point}")
    train_data = [[1,1.5],
                    [1,1]]  # per piazza add in intercepts
    test_data = np.array([1,test_point])  # per piazza add in intercepts

    true_betas = np.array([1,1])
    

    X = np.array(train_data)
    W = np.diag(1 / epsilon_variance)  # piazza

    lambda_arr = []
    bias_sq_arr = []
    var_arr = []
    MSE_arr = []

    for lambda_val in np.linspace(start=.1, stop=5, num=100):  # TA piazza suggestion
        beta_hat = (X.T @ W @ epsilon) * lambda_val
        h_x = true_betas @ X
        ed_f_hat = beta_hat @ X
        ex_f_hat = beta_hat @ test_data
        # print(beta_hat.shape, h_x)

        inner_bias = np.mean(ed_f_hat - h_x)
        bias_sq = inner_bias**2

        beta_hat = np.reshape(beta_hat, (2,1))
        inner_var = np.mean(ex_f_hat - ed_f_hat)
        inner_var = inner_var**2
        MSE = bias_sq + inner_var

        bias_sq_arr.append(bias_sq)
        var_arr.append(inner_var)
        lambda_arr.append(lambda_val)
        MSE_arr.append(MSE)
        
    # # plot MSE (Bias^2 + Variance) vs Lambda
    plt.plot(lambda_arr, bias_sq_arr)
    plt.plot(lambda_arr, var_arr)
    plt.plot(lambda_arr,MSE_arr)
    plt.show()



