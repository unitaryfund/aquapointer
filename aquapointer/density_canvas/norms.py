import numpy as np
import math

def Lp_norm(obj, **kwargs):
    p = kwargs['p']
    return float(abs(obj)**p)

def Lp_distance(obj1, obj2, **kwargs):
    p = kwargs['p']
    return float(abs(obj1-obj2)**p)

def Lp_distance_truncated(obj1, obj2, *kwargs):
    p = kwargs['p']
    high = kwargs['high']
    low = kwargs['low']
    
    coords = obj2._centers
    var = obj2._variance
    amp = obj2._amplitude
    obj1.define_custom_lattice(coords)
    coeffs = obj1.calculate_ubo_coefficients(p, [amp, var], high, low)

    res = 0
    for i in range(low, high+1):
        for val in coeffs[i]:
            res += val
    return res

def KL_distance(canvas1, canvas2, **kwargs):
    if np.linalg.norm(canvas1._origin-canvas2._origin) > 1e-8:
        raise ValueError("The two canvases need to have the same origin")
    if np.abs(canvas1._length_x - canvas2._length_x) > 1e-8:
        raise ValueError("The two canvases need to have the same length on the x-axis")
    if np.abs(canvas1._length_y - canvas2._length_y) > 1e-8:
        raise ValueError("The two canvases need to have the same length on the y-axis")
    if canvas1._npoints_x != canvas2._npoints_x:
        raise ValueError("The two canvases need to have the same number of points on the x-axis")
    if canvas1._npoints_y != canvas2._npoints_y:
        raise ValueError("The two canvases need to have the same number of points on the y-axis")
    
    density1 = canvas1._density
    density2 = canvas2._density

    res = 0
    eps = 1e-15 #regularization
    for i in range(len(density1)):
        for j in range(len(density1[0])):
            p = density1[i,j]
            q = density2[i,j]
            if p<1e-15:
                res += 0
            res += p*np.log(p/(q+eps))
    return res

def JS_distance(canvas1, canvas2, **kwargs):
    return 0.5*KL_distance(canvas1, 0.5*(canvas1+canvas2)) + 0.5*KL_distance(canvas2, 0.5*(canvas1+canvas2)) 