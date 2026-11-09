import numpy as np

def radial_system(X,r,energy,potential,C,l):
  X1,X2 = X
  dX2dr = (((potential(r) - energy)/C) + (l*(l+1))/r**2)*X1
  return np.array([X2,dX2dr])

def radial_initial_conditions(r_min, l):
  return np.array([r_min**(l+1),(l+1)*(r_min**l)])

def radial_residual(r_min,r_max,h,energy,potential,C,l):
  N = int(np.ceil((r_max-r_min)/h)) + 1
  r = np.linspace(r_min,r_max,N)
  h = r[1] - r[0]

  X = np.zeros((N,2))
  X[0] = radial_initial_conditions(r_min,l)

  def f(X,r):
    return radial_system(X,r,energy,potential,C,l)

  for i in range(N-1):
        k1 = h*f(X[i],r[i])
        k2 = h*f(X[i]+0.5*k1,r[i]+0.5*h)
        k3 = h*f(X[i]+0.5*k2,r[i]+0.5*h)
        k4 = h*f(X[i]+k3,r[i]+h)

        X[i+1] = X[i] + (k1+2*k2+2*k3+k4)/6

  residual = X[-1,0]

  return X,r,residual