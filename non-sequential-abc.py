import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns

n = 50 #number of observations
N = 1000 #number of particles

hyper_mu = 0    #mean of the prior 
hyper_sigma = 10 #stddev of the prior

true_mu = 0
sigma = 1


epsilon = 0.005

"""data"""

data = np.random.normal(true_mu , sigma , n)
np.savetxt("data.dat" , data)

"""distance function"""

def rho(y , x):

    return np.abs(np.sum(y) - np.sum(x))/n

"""Rejection Sampling"""
mu = np.zeros(N)
for i in range(N):
   
   d = epsilon + 1
   while (d > epsilon):

       proposed_mu = np.random.normal(hyper_mu , hyper_sigma , 1) #draw from the prior
       x = np.random.normal(proposed_mu , sigma , n)              #forward model
       d = rho(data , x)                                          #distance metric

   mu[i] = proposed_mu
   print i

""" mean and variance of the true posterior"""
post_mu = (hyper_mu / hyper_sigma **2. + np.sum(data) / sigma ** 2.) / (1. / hyper_sigma**2. + n / sigma **2.)
post_sigma = np.sqrt(1. / (1 / hyper_sigma ** 2. + n / sigma ** 2.))

"""plotting the results"""
sns.distplot(mu , 15 , norm_hist = True , label = "Approximate posterior")
mu_range = np.linspace(-1 , 1 , 1000)
post_true = norm(post_mu , post_sigma).pdf(mu_range)
plt.plot(mu_range , post_true , label = "true posterior")
plt.xlabel(r"$\mu$")
plt.legend(loc='upper left')
plt.savefig("/home/mj/public_html/post.png")
