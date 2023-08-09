import numpy as np
import os


# load subspace folder
def subsapce():
    """
    Create and return a subsapce memory.
    """
    Subsapce = {}
    for file in os.listdir("subspace/"): 
        Subsapce[file[:-4]] = np.load("subspace/"+file)
    return Subsapce

Subsapce = subsapce()


for i in range(len(Subsapce.keys())):
    print(Subsapce['subspace'+str(i)].shape)


