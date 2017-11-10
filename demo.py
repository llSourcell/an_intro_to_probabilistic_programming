# import some dependencies
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist


mu = Variable(torch.zeros(1))   # mean zero
sigma = Variable(torch.ones(1)) # unit variance
x = dist.normal(mu, sigma)      # x is a sample from N(0,1)
print(x)

log_p_x = dist.normal.log_pdf(x, mu, sigma)
print(log_p_x)

x = pyro.sample("my_sample", dist.normal, mu, sigma)
print(x)


def weather():
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0], 'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal,
                       Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

for _ in range(3):
    print(weather())
    
    
def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), dist.bernoulli, p)
    if torch.equal(x.data, torch.zeros(1)):
        return x
    else:
        return x + geometric(p, t+1)

print(geometric(Variable(torch.Tensor([0.5]))))
