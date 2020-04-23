# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Data
Table1=pd.read_excel('Data.xlsx',sheet_name='Part Processing Times'index_col=0, header=0)
Table2=pd.read_excel('Data.xlsx',sheet_name='Sequence Dependent Setup Times', index_col=0,header=0)
Table3=pd.read_excel('Data.xlsx',sheet_name='Demand List', index_col=0, header=0)


# SPV method
def sequence(arr):
    seq = np.argsort(arr)
    return seq


# Objective Function
def fitness(arr):
    seq = sequence(arr)
    wip = 2
    y = np.zeros([n, n]) a = -2
    while a < len(seq) - 1:
        y[seq[a]][seq[a + 1]] = 1
        a += 1
    series = 0
    for p in range(0, len(T)):
        for q in range(0, len(T[p])):
            series += wip * (y[p][q] * T[p][q])
    time = 0
    for p in range(0, m):
        for q in range(0, n):
            for work in range(0, wip):
                if d[q] % wip == work:
                    time += P[p][q] * (d[q] + work) / wip
    makespan = time + series
    return makespan


# Variables Initialization
n = len(Table3.columns)	# dimension for PSO
m = len(Table1)
d = Table3.to_numpy()[0]
T = Table2.to_numpy()
P = Table1.to_numpy()
VarMin = 0	# Lower Bound
VarMax = 4.0	# Upper Bound
MaxVel = 1 * (VarMax - VarMin)
MinVel = -MaxVel
GlobalBestTime = 10000	# Random Large Number
RecordedBestTime = []

# Clerc and Kennedy (2002) Constriction Coefficient k = 1
phi_1 = 2.05
phi_2 = 2.05
phi = phi_1 + phi_2
chi = (2 * k) / (np.abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi)))

# PSO parameters
global nPop
global Maxit
Maxit = 100  # Maximum iteration
nPop = 10	# Population size

w = chi  # Inertia coefficient
damp = 1  # Damping ratio of inertia
c1 = chi * phi_1  # Personal acceleration coefficient
c2 = chi * phi_2	# Social acceleration coefficient


# Template
class EmptyParticle:
    def init(self):
        self.position = np.array([])
        self.velocity = np.array([])
        self.time = np.array([])
        self.best = self.Best()

    class Best:
        def init(self):
            self.position = np.array([])
            self.time = np.array([])

@classmethod
def reshape(cls, self):
    global VarMin
    global VarMax
    global GlobalBestTime
    global GlobalBestPosition
    global n
    for i in range(0, n):
        # Generate Random Solution
        self.position = np.append(self.position, np.random.uniform(VarMin, VarMax))
        self.velocity = np.append(self.velocity, np.random.uniform(MaxVel, MinVel))
    # Evaluation of Time
    self.time = fitness(self.position)
    # Update Personal Best
    self.best.position = self.position
    self.best.time = self.time
    # Update Global Best
    if self.best.time < GlobalBestTime:
        GlobalBestTime = self.best.time
        GlobalBestPosition = self.best.position

# PSO Initialization
particle = [EmptyParticle() for i in range(0, nPop)]
for i in range(0, nPop):
    EmptyParticle.reshape(particle[i])

# Plotting Results
def plot(arr):
    plt.semilogy(arr)
    plt.grid(b=bool, which="both", axis="both", color='g', linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan Time")
    plt.title("PSO Implementation in Python")
    plt.savefig('Solution.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

# Main Loop
order = []
for i in range(0, Maxit):
    for j in range(0, nPop):
        # Update Velocity
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        for k in range(n):
            a_1 = (particle[j].best.position[k] - particle[j].position[k])
            a_2 = (GlobalBestPosition[k] - particle[j].position[k])
            particle[j].velocity[k] = w * particle[j].velocity[k] + c1 * r1 * a_1 + c2 * r2 * a_2
        # Apply Limits to Velocity
        particle[j].velocity = np.maximum(particle[j].velocity, MinVel)
        particle[j].velocity = np.minimum(particle[j].velocity, MaxVel)
        # Update Position
        for k in range(n):
            particle[j].position[k] = particle[j].position[k] + particle[j].velocity[k]
        # Apply Limits to Position
        particle[j].position = np.maximum(particle[j].position, VarMin)
        particle[j].position = np.minimum(particle[j].position, VarMax)
        # Evaluation of Time
        particle[j].time = fitness(particle[j].position)
        # Update Personal Best
        if particle[j].time < particle[j].best.time:
            particle[j].best.time = particle[j].time
            particle[j].best.position = particle[j].position
            order = sequence(particle[j].position)
        # Update Global Best
        if particle[j].best.time < GlobalBestTime:
            GlobalBestTime = particle[j].best.time
            GlobalBestPosition = particle[j].best.position
    # BestTime
    print("Iteration: %d" % i, " Best Time: ", GlobalBestTime, " Job Sequence: ", order)
    RecordedBestTime.append(GlobalBestTime)
    # Damping Inertia w = w * damp
plot(RecordedBestTime)  # Calling Plot Function