import numpy as np
import matplotlib.pyplot as plt

def quantum_system(X,xi,energy,potential):
  psi,dpsi = X
  potential_val = potential(xi)

  return np.array([dpsi,(potential_val - energy)*psi])

def residual(potential,xi_initial,xi_final,energy,h):
  xi_values = np.arange(xi_initial, xi_final + h, h)
  N = len(xi_values)

  X = np.zeros((N, 2))
  X[0] = [0.0, 1.0]

  def f(X,xi):
    return quantum_system(X,xi,energy,potential)

  for i in range(N-1):
      k1 = h*f(X[i],xi_values[i])
      k2 = h*f(X[i]+0.5*k1,xi_values[i]+0.5*h)
      k3 = h*f(X[i]+0.5*k2,xi_values[i]+0.5*h)
      k4 = h*f(X[i]+k3,xi_values[i]+h)

      X[i+1] = X[i] + (k1+2*k2+2*k3+k4)/6

  residual = X[-1, 0]

  return X, xi_values,residual

def quantum_shooting(potential,xi_initial,xi_final,energy_low,energy_high,h,energy_tol,residual_tol,max_iter):
  X, xi_values,residual_low = residual(potential,xi_initial,xi_final,energy_low,h)
  X, xi_values,residual_high = residual(potential,xi_initial,xi_final,energy_high,h)

  error = abs(energy_high - energy_low)
  itr   = 0
  if residual_low*residual_high>0:
    raise ValueError("No sign change in the given interval")

  elif abs(residual_low) < residual_tol:
    return energy_low

  elif abs(residual_high) < residual_tol:
    return energy_high

  else:
    while error > energy_tol and itr < max_iter:
      energy_mid   = (energy_low + energy_high)/2
      X, xi_values,residual_mid = residual(potential,xi_initial,xi_final,energy_mid,h)

      if abs(residual_mid) < residual_tol:
        return energy_mid

      elif residual_low*residual_mid < 0:
        residual_high = residual_mid
        energy_high = energy_mid
        error = abs(energy_high - energy_low)

      elif residual_high*residual_mid < 0:
        residual_low = residual_mid
        energy_low = energy_mid
        error = abs(energy_high - energy_low)
      itr += 1


  return (energy_high + energy_low)/2


def wavefunction(potential, xi_initial, xi_final, energy, h):
    X, xi_values, residual_value = residual(
        potential, xi_initial, xi_final, energy, h
    )

    psi = X[:, 0]

    I = np.trapezoid(np.abs(psi)**2, xi_values)
    psi_normalized = psi / np.sqrt(I)

    return xi_values, psi, psi_normalized



def plot_wavefunction(psi, psi_normalized, energy, xi_values, potential_name):

    area = np.trapezoid(np.abs(psi)**2, xi_values)
    area_normalized = np.trapezoid(np.abs(psi_normalized)**2, xi_values)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].plot(xi_values, psi, label=f"Integral = {area:.6f}")
    ax[0].set_title("Non-normalized wavefunction")
    ax[0].set_xlabel(r"$\xi$")
    ax[0].set_ylabel(r"$\psi(\xi)$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(xi_values, psi_normalized,label=f"Integral = {area_normalized:.6f}")
    ax[1].set_title("Normalized wavefunction")
    ax[1].set_xlabel(r"$\xi$")
    ax[1].set_ylabel(r"$\psi(\xi)$")
    ax[1].legend()
    ax[1].grid()

    plt.suptitle(f"{potential_name} (Energy = {energy:.6f})")
    plt.tight_layout()
    plt.show()
