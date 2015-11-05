import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import matplotlib
import plotting
from  multiprocessing import Pool
from scipy.stats import norm , gamma
plotting.prettyplot()
import seaborn as sns
import matplotlib.cm as cm
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

def sample_posterior(Nsample):
    
    xbar , s = np.mean(data) , np.var(data)
    hyper_t = 1./hyper_sigma**2.
    a_pos = n/2. + hyper_a
    scale_pos  = 1./(1 + n*s/2.)
    
    m = np.zeros((Nsample))
    t = np.zeros((Nsample))
    
    for i in range(Nsample):
   
        t[i] = gamma.rvs(a_pos, loc=0, scale=scale_pos) 
        sigma_pos = (n*t[i] + hyper_t)**-.5
        mean_pos = (n*xbar*t[i] + hyper_mu*hyper_t)/(n*t[i] + hyper_t)
        m[i] = norm.rvs(loc = mean_pos, scale=sigma_pos)
    
    return m, t
    
m , t =sample_posterior(3000)    
sns.jointplot(m, t, kind="kde", xlim=[m.min(),m.max()], ylim=[t.min(),t.max()],
              joint_kws={"gridsize":30, "cmap":cm.BuPu,
                        "extent":[m.min(),m.max(),t.min(),t.max()]})
plt.savefig("/home/mj/public_html/test-abc/2d/true.png")
plt.close()

mm , tt  = np.loadtxt("2d-mu.dat") , np.loadtxt("2d-tau.dat")
print mm.shape
#sns.jointplot(m, t, kind="kde", xlim=[mm.min(),mm.max()], ylim=[tt.min(),tt.max()],
#              joint_kws={"gridsize":30, "cmap":cm.BuPu,
#                        "extent":[mm.min(),mm.max(),tt.min(),tt.max()]})
plt.scatter(mm, tt)
plt.savefig("/home/mj/public_html/test-abc/2d/approx-true-2d.png")
plt.close()

sns.distplot(mm , 11 , norm_hist = True , kde = False, label = "Approximate Posterior")
sns.distplot(m ,  51 , norm_hist = True , hist = False, kde = True, label = "True Posterior")
plt.xlabel(r"$\mu$")
plt.savefig("/home/mj/public_html/test-abc/2d/true-approx-mean.png")
plt.close()



sns.distplot(tt , 11 , norm_hist = True , kde = False, label = "Approximate Posterior")
sns.distplot(t ,  51 , norm_hist = True , hist = False, kde = True, label = "True Posterior")
plt.xlabel(r"$\tau$")
plt.savefig("/home/mj/public_html/test-abc/2d/true-approx-tau.png")
plt.close()
