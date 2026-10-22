"""Fixed-sample episode statistics; no pooling of seen and unseen strata."""
import math
from scipy.stats import fisher_exact


def wilson(k,n):
    if not isinstance(k,int) or not isinstance(n,int) or not 0<=k<=n or n<1:raise ValueError('integer binomial count 0<=k<=n, n>0 required')
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def compare(a,na,b,nb):
    ca=wilson(a,na);cb=wilson(b,nb);p=float(fisher_exact([[a,na-a],[b,nb-b]],alternative='two-sided')[1])
    return dict(a=dict(k=a,n=na,wilson95=ca),b=dict(k=b,n=nb,wilson95=cb),fisher_two_sided_p=p,
                registered_a_superior=bool(a/na>b/nb and p<.05 and ca[0]>cb[1]))
