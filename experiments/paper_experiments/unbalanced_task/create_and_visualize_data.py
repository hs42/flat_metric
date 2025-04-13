import numpy as np
import matplotlib.pyplot as plt


sigma1 = 0.3
cov1 = np.array([[sigma1, 0],
                [0, sigma1]])
cov2 = np.array([[5*sigma1, 0],
                [0, sigma1]])
cov3 = sigma1 * np.array([[1.0, 0.5],
                [0.5, 1.0]]) 
cov4 = sigma1 * np.array([[1.5, 1.0],
                [1.0, 1.5]]) 
cov5 = 2 * cov1

N_A = 50
N_B = N_A
N_C = 150

# before, i.e. source distribution

A = np.random.multivariate_normal((1,5), cov1, N_A)
B = np.random.multivariate_normal((5,3.5), cov1, N_B)
C = np.random.multivariate_normal((3,1), cov2, N_C)

np.savetxt('A_before', A)
np.savetxt('B_before', B)
np.savetxt('C_before', C)

plt.scatter(A[:,0], A[:,1], label='A')
plt.scatter(B[:,0], B[:,1], label='B')
plt.scatter(C[:,0], C[:,1], label='C')

plt.xlabel('feature 1', size = 18)
plt.ylabel('feature 2', size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.xlim([-1,8])
plt.ylim([-1,7])
plt.legend(fontsize=15)
plt.savefig('Fig15.eps', format='EPS', bbox_inches='tight')

# after, i.e. target distribution
N_A = 38
N_B = 42
N_C = 113

A = np.random.multivariate_normal((2,5), cov3, N_A)
B = np.random.multivariate_normal((5,3.5), cov4, N_B)
C = np.random.multivariate_normal((6.5,1), cov5, N_C)

np.savetxt('A_after', A)
np.savetxt('B_after', B)
np.savetxt('C_after', C)

plt.clf()
plt.scatter(A[:,0], A[:,1], label='X', c='yellow', marker='x')
plt.scatter(B[:,0], B[:,1], label='Y', c='magenta', marker='x')
plt.scatter(C[:,0], C[:,1], label='Z', c='black', marker='x')

plt.xlabel('feature 1', size = 18)
plt.ylabel('feature 2', size = 18)
plt.xticks(size = 18)
plt.yticks(size = 18)
plt.xlim([-1,8])
plt.ylim([-1,7])
plt.legend(fontsize=15)
plt.savefig('Fig16.eps', format='EPS', bbox_inches='tight')