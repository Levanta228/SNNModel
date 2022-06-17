# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nengo

tau = 0.1
sigma = 10
beta = 8.0 / 3
rho = 28


def feedback(x):
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho

    return [
        dx0 * tau + x[0],
        dx1 * tau + x[1],
        dx2 * tau + x[2],
    ]


model = nengo.Network(label="Lorenz attractor")
with model:
    state = nengo.Ensemble(2000, 3, radius=60)
    nengo.Connection(state, state, function=feedback, synapse=tau)
    state_probe = nengo.Probe(state, synapse=tau)

with nengo.Simulator(model) as sim:
    sim.run(10)

ax = plt.figure().add_subplot(111, projection=Axes3D.name)
ax.plot(*sim.data[state_probe].T)

plt.figure()
plt.plot(sim.trange(), sim.data[state_probe])
plt.show()
