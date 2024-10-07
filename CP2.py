import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import minimize

movie_index = np.array(range(1, 26))
ratings = np.array([7.6, 8.0, 8.0, 8.1, 7.8, 7.7, 7.8, 7.5, 8.3, 7.4, 8.6, 7.5, 7.4, 8.0, 8.2, 7.6, 7.8, 8.2, 7.6, 7.6, 7.5, 7.4, 7.7, 7.9, 7.7])

def linear_model(params, x):
    b0, b1 = params
    return b0 + b1 * x

# Residuals function
def residuals(params, y, x):
    return y - linear_model(params, x)

# Initial guess for parameters
initial_params = [1.0, 1.0]

# Fit the model using least squares
result_ls = leastsq(residuals, initial_params, args=(ratings, movie_index))

# Extract the optimal parameters
beta0_ls, beta1_ls = result_ls[0]

# Print the results
print("Least Squares Results:")
print(f"Intercept (beta0): {beta0_ls}")
print(f"Slope (beta1): {beta1_ls}")

# Constrained linear model function
def constrained_linear_model(params, x):
    b0, b1 = params
    return b0 + b1 * x

# Objective function to minimize
def objective(params, y, x):
    return np.sum((y - constrained_linear_model(params, x))**2)

# Constraint function: beta1 >= 0
def constraint(params):
    return params[1]  # beta1

# Initial guess for parameters
initial_params_cls = [1.0, 1.0]

# Define the constraint
cons = {'type': 'ineq', 'fun': constraint}

# Minimize the objective function subject to the constraint
result_cls = minimize(objective, initial_params_cls, args=(ratings, movie_index), constraints=cons)

# Extract the optimal parameters
beta0_cls, beta1_cls = result_cls.x

# Print the results
print("\nConstrained Least Squares Results:")
print(f"Intercept (beta0): {beta0_cls}")
print(f"Slope (beta1): {beta1_cls}")

X = np.vstack([np.ones_like(movie_index), movie_index]).T

# Solve the LS problem using the inverse
beta_ls_inv = np.linalg.inv(X.T @ X) @ X.T @ ratings

# Print the results
print("\nInverse of Corresponding Matrix - Least Squares Results:")
print(f"Intercept (beta0): {beta_ls_inv[0]}")
print(f"Slope (beta1): {beta_ls_inv[1]}")

# Solve the LS problem using QR factorization with classic Gram-Schmidt method
Q, R = np.linalg.qr(X)
beta_ls_qr_classic = np.linalg.solve(R, Q.T @ ratings)

# Print the results
print("\nQR Factorization with Classic Gram-Schmidt - Least Squares Results:")
print(f"Intercept (beta0): {beta_ls_qr_classic[0]}")
print(f"Slope (beta1): {beta_ls_qr_classic[1]}")

# Modified Gram-Schmidt method
def qr_modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].T, v)
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

# Solve the LS problem using QR factorization with modified Gram-Schmidt method
Q_mod, R_mod = qr_modified_gram_schmidt(X)
beta_ls_qr_modified = np.linalg.solve(R_mod, Q_mod.T @ ratings)

print("\nQR Factorization with Modified Gram-Schmidt - Least Squares Results:")
print(f"Intercept (beta0): {beta_ls_qr_modified[0]}")
print(f"Slope (beta1): {beta_ls_qr_modified[1]}")

movie_index_26th = 26

# Predict the rating using the fitted models
predicted_rating_ls = linear_model([beta0_ls, beta1_ls], movie_index_26th)
predicted_rating_cls = constrained_linear_model([beta0_cls, beta1_cls], movie_index_26th)
predicted_rating_inv = linear_model(beta_ls_inv, movie_index_26th)
predicted_rating_qr_classic = linear_model(beta_ls_qr_classic, movie_index_26th)
predicted_rating_qr_modified = linear_model(beta_ls_qr_modified, movie_index_26th)

# predicted ratings
print("Predicted Ratings for the 26th Movie:")
print(f"Least Squares: {predicted_rating_ls}")
print(f"Constrained Least Squares: {predicted_rating_cls}")
print(f"Inverse of Corresponding Matrix: {predicted_rating_inv}")
print(f"QR Factorization - Classic Gram-Schmidt: {predicted_rating_qr_classic}")
print(f"QR Factorization - Modified Gram-Schmidt: {predicted_rating_qr_modified}")

extended_movie_index = np.arange(1, 27)

#  data points
plt.scatter(movie_index, ratings, label='Data Points')

#  least squares
plt.plot(extended_movie_index, linear_model([beta0_ls, beta1_ls], extended_movie_index), label='Least Squares', linestyle='--')

#  constrained least squares
plt.plot(extended_movie_index, constrained_linear_model([beta0_cls, beta1_cls], extended_movie_index), label='Constrained Least Squares', linestyle='--')

# inverse of the matrix
plt.plot(extended_movie_index, linear_model(beta_ls_inv, extended_movie_index), label='Inverse of Corresponding Matrix', linestyle='--')

#  fitted line from QR factorization with classic Gram-Schmidt
plt.plot(extended_movie_index, linear_model(beta_ls_qr_classic, extended_movie_index), label='QR Factorization - Classic Gram-Schmidt', linestyle='--')

# fitted line from QR factorization with modified Gram-Schmidt
plt.plot(extended_movie_index, linear_model(beta_ls_qr_modified, extended_movie_index), label='QR Factorization - Modified Gram-Schmidt', linestyle='--')

plt.xlabel('Movie Index')
plt.ylabel('Ratings')
plt.title('Movie Ratings and Fitted Lines')
plt.legend()

# visualization
plt.show()