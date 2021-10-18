import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, std=1, lower=0, upper=10):
    return truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()