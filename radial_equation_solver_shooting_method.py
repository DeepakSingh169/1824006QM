import numpy as np

def radial_system(X,r,energy,potential,C,l):
  X1,X2 = X
  dX2dr = (((potential(r) - energy)/C) + (l*(l+1))/r**2)*X1
  return np.array([X2,dX2dr])

def radial_initial_conditions(r_min, l):
  return np.array([r_min**(l+1),(l+1)*(r_min**l)])