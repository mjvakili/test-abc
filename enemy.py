import abcpmc
from scipy.stats import norm , gamma
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
sns.set_style("white")
#rc('text', usetex=True)
#rc('axes', labelsize=15, titlesize=15)
import numpy as np
np.random.seed()


data = np.loadtxt("2d-data.dat")
n = 50 #number of observations
N = 5000 #number of particles

""" hyper parameters of the prior over mean and precision """
hyper_mu =  0.    #mean of the prior over mean 
hyper_sigma = 10. #stddev of the prior over mean
hyper_a = 1.      #parameter of the prior over precision

""" true mean and precision of the distribution N(\mu , \tau^-1) that n i.i.d rvs are drawn from"""
true_mu = 0.      #true mean
true_tau = 1.

def dist_measure(x, y):
    return np.abs(np.mean(x) - np.mean(y)) + np.abs(np.std(x) - np.std(y))


def create_new_sample(theta):
    return np.random.normal(theta[0], theta[1]**-.5, n)


class Prior(object):
    
    def __init__(self, hyper_mu, hyper_sigma, hyper_a):
        self.hyper_mu = hyper_mu
        self.hyper_sigma = hyper_sigma
        self.hyper_a = hyper_a
        
    def __call__(self, theta=None):
        if theta is None:
            return np.array([norm.rvs(self.hyper_mu , self.hyper_sigma , 1)[0] , gamma.rvs(self.hyper_a , size = 1)[0]])
        else:
            return norm.pdf(theta[0], self.hyper_mu, self.hyper_sigma)*gamma.pdf(theta[1] , self.hyper_a)


prior = Prior(hyper_mu=0., hyper_sigma=10., hyper_a=1.)

#print prior([1. , 1.])
#print prior()


alpha = 75
T = 26
eps_start = 1.0
eps = abcpmc.ConstEps(T, eps_start)

sampler = abcpmc.Sampler(N=5000, Y=data, postfn=create_new_sample, dist=dist_measure, threads=20)

sampler.particle_proposal_cls = abcpmc.OLCMParticleProposal

def launch():
    eps = abcpmc.ConstEps(T, eps_start)

    pools = []
    for pool in sampler.sample(prior, eps):
        print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, eps(pool.eps), pool.ratio))
        np.savetxt("theta"+str(pool.t)+".dat" , pool.thetas)
        np.savetxt("weights"+str(pool.t)+".dat" , pool.ws)
        
        for i, (mean, std) in enumerate(zip(*abcpmc.weighted_avg_and_std(pool.thetas, pool.ws, axis=0))):
            print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))

        eps.eps = np.percentile(pool.dists, alpha) # reduce eps value
        pools.append(pool)
    sampler.close()
    return pools


import time
t0 = time.time()
pools = launch()
print "took", (time.time() - t0)


