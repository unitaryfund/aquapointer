import numpy as np
import math

def Lp_norm(obj, p):
    return float(abs(obj)**p)

def Lp_distance(obj1, obj2, p):
    return float(abs(obj1-obj2)**p)

def KL_distance(canvas1, canvas2):
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

def JS_distance(canvas1, canvas2):
    return 0.5*KL_distance(canvas1, 0.5*(canvas1+canvas2)) + 0.5*KL_distance(canvas2, 0.5*(canvas1+canvas2)) 