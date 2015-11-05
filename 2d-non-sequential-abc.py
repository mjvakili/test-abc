import numpy as np
from scipy.stats import norm , gamma
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend("Agg")
import plotting
from  multiprocessing import Pool

plotting.prettyplot()
#import seaborn as sns

n = 50 #number of observations
N = 2000 #number of particles

""" hyper parameters of the prior over mean and precision """
hyper_mu =  0.    #mean of the prior over mean 
hyper_sigma = 10. #stddev of the prior over mean
hyper_a = 1.      #parameter of the prior over precision

""" true mean and precision of the distribution N(\mu , \tau^-1) that n i.i.d rvs are drawn from"""
true_mu = 0.      #true mean
true_tau = 1.     #true precision

"""data"""

#data = np.random.normal(true_mu , true_tau**-.5 , n)
#np.savetxt("2d-data.dat" , data)
data = np.loadtxt("2d-data.dat")

"""true posterior"""

def posterior(m , t):
    
    xbar , s = np.mean(data) , np.var(data)
    hyper_t = 1./hyper_sigma**2.
   
    sigma_pos = (n*t + hyper_t)**-.5
    mean_pos = (n*xbar*t + hyper_mu*hyper_t)/(n*t + hyper_t)

    a_pos = n/2. + hyper_a
    scale_pos  = 1./(1 + n*s/2.)
    #print mean_pos
    #print sigma_pos
    return gamma.pdf(t , a_pos , loc=0 , scale=scale_pos) * norm.pdf(m , loc = mean_pos , scale=sigma_pos)


    
m = np.linspace(-1., 1., 1000)
t = np.linspace(.25, 2., 1000)
M, T = np.meshgrid(m, t)
P = posterior(M,T)

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
CS = plt.contour(M, T, P)
plt.colorbar(CS)
plt.xlim((-1.,1.))
plt.ylim((.25,2.))
plt.savefig("/home/mj/public_html/2d-pos-true.png")

#print true_posterior


""" rejection threshold """
epsilon = 0.015

"""distance function"""

def rho(y , x):

    return np.abs(np.mean(y) - np.mean(x)) + np.abs(np.std(y) - np.std(x))

"""Rejection Sampling"""

mu = np.zeros(N)
tau = np.zeros(N)

def sampler(i):
   
   d = epsilon + 1
   while (d > epsilon):

       proposed_mu = np.random.normal(hyper_mu , hyper_sigma , 1) #draw from the prior over the mean
       proposed_tau = gamma.rvs(1. , size = 1)                    #draw from the prior over precision
       x = np.random.normal(proposed_mu , proposed_tau**-.5 , n)  #forward model
       d = rho(data , x)                                          #distance metric
   #print i
   return proposed_mu , proposed_tau

def abc():

    pool = Pool(processes=20)
    mapfn = pool.map
    results = mapfn(sampler , np.arange(N))
    pool.close()
    pool.terminate()
    pool.join()
    results = np.array(results)[:,:,0]
    print results.shape    
    mu , tau = results[:,0] , results[:,1]

    return mu , tau

mu , tau = abc()    

np.savetxt("2d-mu.dat" , mu)
np.savetxt("2d-tau.dat", tau)

""" mean and variance of the true posterior"""
#post_mu = (hyper_mu / hyper_sigma ** 2. + np.sum(data) / sigma ** 2.) / (1. / hyper_sigma**2. + n / sigma **2.)
#post_sigma = np.sqrt(1. / (1 / hyper_sigma ** 2. + n / sigma ** 2.))

"""plotting the results"""
"""
sns.distplot(mu , 15 , norm_hist = True , label = "Approximate posterior")
mu_range = np.linspace(-1 , 1 , 1000)
post_true = norm(post_mu , post_sigma).pdf(mu_range)
plt.plot(mu_range , post_true , label = "true posterior")
plt.xlabel(r"$\mu$")
plt.legend(loc='upper left')
plt.savefig("/home/mj/public_html/post.png")
"""
import pandas as pd
import seaborn as sns
sns.set(style="white")

# Generate a random correlated bivariate dataset
#rs = np.random.RandomState(5)
#mean = [0, 0]
#cov = [(1, .5), (.5, 1)]
#x1, x2 = rs.multivariate_normal(mean, cov, 500).T

x1 = pd.Series(mu, name=r"$\mu$")
x2 = pd.Series(tau, name=r"$\tau$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(x1, x2, kind="kde", size=7, space=0)
plt.savefig("/home/mj/public_html/2dpos_05.png")

