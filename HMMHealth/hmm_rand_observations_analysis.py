"""
Gaussisan HMM of data generated randomly
Uses Gaussian HMM on data generated randomly
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


# Generate random numbers for observations. Each observation has 2 attributes
X=np.random.rand(100,2)

# Run Gaussian HMM
print "fitting these numbers to HMM"

#Create an HMM instance with 4 components and using diagonal covariance
model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

# Print trained parameters
print "Transition matrix"
print model.transmat_

print "Means and vars of each hidden state"
for i in range(model.n_components):
    print "{0}th hidden state".format(i)
    print "mean = ", model.means_[i]
    print "var = ", np.diag(model.covars_[i])
