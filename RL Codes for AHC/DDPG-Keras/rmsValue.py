

                        ## For calculating the RMS value of an array ##

import math

def rmsValue(array):
    n = len(array)
    squre = 0.0
    root = 0.0
    mean = 0.0
    
    # Calculating Squre
    for i in range(0, n):
        squre += (array[i] ** 2)
    # Calculating Mean
    mean = (squre/ (float)(n))
    # Calculating Root
    root = math.sqrt(mean)
    return root