import numpy as np

def Lp_norm(obj, p):
    return float(abs(obj)**p)

def Lp_distance(obj1, obj2, p):
    return float(abs(obj1-obj2)**p)

def KL_norm(obj1, obj2):
    #to be implemented
    return 0

def JS_norm(obj1, obj2):
    #to be implemented
    return 0