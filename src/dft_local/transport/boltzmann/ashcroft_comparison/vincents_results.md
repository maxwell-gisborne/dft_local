'''
    Constants
# --------------------------
e = 1.602176634e-19       # C
hbar = 1.054571817e-34    # J*s
kB = 1.380649e-23         # J/K
hartree_to_joule = 4.3597447222071e-18  # 1 Hartree in J
bohr_to_m = 0.52917721092e-10  # 1 Bohr in meters

# --------------------------
# Material parameters
# --------------------------
T = 300                   # Temperature (K)
tau = 1e-14               # Relaxation time

E_vec = np.array([1e5, 0.0])  # V/m

mu  is set to avereage_epsilon (in JOules) other the fermi function is basically 0.                  # Chemical potential (J)


ample velocities at some k-points:
k[0] = [ 0. -0.], v = [ -7090.46879031 -12102.11158297] m/s
k[1] = [ 2.55464699e+08 -1.47492613e+08], v = [ 14075.04503413 -24202.4466007 ] m/s
k[2] = [ 5.10929398e+08 -2.94985226e+08], v = [ 28253.4799914  -24202.44660072] m/s
k[3] = [ 7.66394098e+08 -4.42477839e+08], v = [ 49414.51560173 -36301.36263201] m/s
k[4] = [ 1.02185880e+09 -5.89970451e+08], v = [ 77551.52350077 -60490.82133335] m/s

Shifted k-point: [        0.         -29498522.56891833], v_shift = [ -7090.46879031 -12102.11158297] m/s

Fermi factor statistics:
Max f*(1-f) = 2.499e-01
Min f*(1-f) = 0.000e+00
Mean f*(1-f) = 3.907e-03

Total runtime: 38.91 s
Conductivity tensor σ_αβ [S/m]:
[[ 6.45179383e-02 -8.80479820e-05]
 [-8.73823365e-05  6.44024548e-02]]
'''
