"""Gaussian option control-variate algebra, with an essential analytic correction."""
import numpy as np


def raw_gradients(reward,baseline,epsilon,derivative,sigma=1.):
    score=epsilon/(sigma*sigma)
    return ((reward-baseline-(derivative*epsilon).sum(-1))[:,None]*score+derivative)


def numpy_reverse(parameters,inputs,output_gradient,activation=None):
    x=np.asarray(inputs,np.float64);tape=[]
    layers=sorted(int(k.split('.')[0]) for k in parameters if k.endswith('.weight'))
    for j,layer in enumerate(layers):
        weight=parameters[str(layer)+'.weight'].numpy().astype(np.float64)
        z=x@weight.T+parameters[str(layer)+'.bias'].numpy();tape.append((weight,z))
        x=np.maximum(z,0) if j<len(layers)-1 else z
    cot=np.asarray(output_gradient,np.float64)
    if activation=='sigmoid':
        output=1/(1+np.exp(-x.clip(-700,700)));cot=cot*output*(1-output)
    for j in range(len(tape)-1,-1,-1):
        cot=cot@tape[j][0]
        if j:cot=cot*(tape[j-1][1]>0)
    return cot
