import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import seaborn as sns

n = 25
N = 1000

hyper_mu = 0
hyper_sigma = 10

true_mu = 0.
sigma = 1. 

epsilon0 = 1.

T = 20

data = np.random.normal(true_mu , sigma , n)
np.savetxt("data.dat" , data)



weights = np.ones((T , N))*1./N
mu      = np.zeros((T , N))
d       = np.zeros((T , N))
epsilon = np.ones(T)*epsilon0

def rho(y, x):

    return np.abs(np.sum(y) - np.sum(x))/n


post_mu = (hyper_mu / hyper_sigma **2. + np.sum(data) / sigma ** 2.) / (1. / hyper_sigma**2. + n / sigma **2.)
post_sigma = np.sqrt(1. / (1 / hyper_sigma ** 2. + n / sigma ** 2.))          
                                                                              
def plot(t , mu , weights):
    sns.distplot(mu[t] , 21 , norm_hist = True , label = "Approximate Posterior")    
    mu_range = np.linspace(-1 , 1 , 1000)
    post_true = norm(post_mu , post_sigma).pdf(mu_range)
    prior= norm(hyper_mu , hyper_sigma).pdf(mu_range)
    plt.plot(mu_range , post_true , label = "true posterior")
    #plt.plot(mu_range , prior , label = "prior")                     
    plt.xlabel(r"$\mu$")
    plt.legend(loc='upper left')
    plt.savefig("/home/mj/public_html/post_pmc"+str(t)+".png")
    plt.close()

def weighted_plot(t , mu , weights):

    sns.distplot(mu[t] , 21 , norm_hist = True , hist_kws={'weights': weights[t]} , label = "Approximate Posterior")    
    mu_range = np.linspace(-1 , 1 , 1000)
    post_true = norm(post_mu , post_sigma).pdf(mu_range)
    prior= norm(hyper_mu , hyper_sigma).pdf(mu_range)
    plt.plot(mu_range , post_true , label = "true posterior")                     
    #plt.plot(mu_range , prior , label = "prior")                     
    plt.xlabel(r"$\mu$")
    plt.legend(loc='upper left')
    plt.savefig("/home/mj/public_html/post_pmc_weighted"+str(t)+".png")
    plt.close()


for t in range(T):

   if (t == 0):
      for i in range(N):
         
	 d[t, i] = epsilon[t] + 1.
         while (d[t,i] > epsilon[t]):
             np.random.seed()
             proposed_mu = np.random.normal(hyper_mu, hyper_sigma, 1)

  	     x = np.random.normal(proposed_mu, sigma, n)
             d[t,i] = rho(data, x)
         mu[t,i] = proposed_mu

   else:
      epsilon[t] = np.percentile(d[t-1], 75)
      mean_prev = np.sum(mu[t-1, :]*weights[t-1,:])
      var_prev  = np.sum((mu[t-1, :] - mean_prev)**2. * weights[t-1 , :])
      #var_prev  = np.cov(mu[t-1])
      for i in range(N):

         d[t,i] = epsilon[t] + 1
         while (d[t,i]>epsilon[t]):
	     np.random.seed()
             index = np.random.choice(np.arange(N), 1, p=weights[t-1])[0]
             proposed_mu0 = mu[t-1, index]
	     proposed_mu  = np.random.normal(proposed_mu0, np.sqrt(2.*var_prev), 1)
             x = np.random.normal(proposed_mu , sigma , n)         
             d[t,i] = rho(data , x)
         mu[t,i] = proposed_mu
         mu_weight_denominator = np.sum(weights[t-1, :]*norm(mu[t-1,:] , np.sqrt(2. * var_prev)).pdf(proposed_mu))
         mu_weight_numerator   = norm(hyper_mu , hyper_sigma).pdf(proposed_mu)
         weights[t,i] = mu_weight_denominator / mu_weight_numerator
   
   weights[t] = weights[t] / np.sum(weights[t])
   plot(t, mu , weights)
   weighted_plot(t , mu , weights)
   print epsilon[t]

