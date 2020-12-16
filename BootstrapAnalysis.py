# Licensed under the MIT License <http://opensource.org/MIT>
#
# Copyright (c) 2020 Shimpei Goto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnishedto do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import toml
import json
from glob import glob
plt.style.use('seaborn-ticks')
plt.style.use('favorite')


def AddData(directory, storage):
    energies = []
    samples = glob(directory+'/sample_*.msg')
    norm_sq_arr = []
    ene_arr = []
    ene_sq_arr = []
    Nsample = 0
    for i, sample in enumerate(samples):
        data = json.load(open(sample))
        if i == 0:
            if 'beta' not in storage:
                storage['beta'] = np.array(data['beta'])
        Nsample += len(data['Samples'])
        energies.append(data['LowestEnergy'])

    gene = min(energies)

    for sample in samples:
        data = msgpack.unpackb(open(sample, 'rb').read(), raw=False)
        for each in data['Samples']:
            norm = (np.exp(0.5*storage['beta']*(gene - each['Energy'][-1]))
                    * np.array(each['Norm']))
            ene = np.array(each['Energy'])
            ene_sq = np.array(each['SquaredEnergy'])
            norm_sq_arr.append(norm*norm)
            ene_arr.append(norm*norm*ene)
            ene_sq_arr.append(norm*norm*ene_sq)
    norm_sq_arr = np.array(norm_sq_arr).T
    ene_arr = np.array(ene_arr).T
    ene_sq_arr = np.array(ene_sq_arr).T
    storage['LowestEnergy'].append(gene)
    storage['SquaredNorm'].append(norm_sq_arr)
    storage['Energy'].append(ene_arr)
    storage['SquaredEnergy'].append(ene_sq_arr)
    storage['Nsamples'].append(Nsample)


def Bootstrapped(storage):
    bootstrapped = {'SquaredNorm': [], 'Energy': [], 'SquaredEnergy': []}
    nset = len(storage['LowestEnergy'])
    for n in range(nset):
        nsample = storage['Nsamples'][n]
        select = np.random.randint(nsample, size=nsample)
        bootstrapped['SquaredNorm'].append(
                np.average(storage['SquaredNorm'][n][:, select], axis=1)
                )
        bootstrapped['SquaredEnergy'].append(
                np.average(storage['SquaredEnergy'][n][:, select], axis=1)
                )
        bootstrapped['Energy'].append(
                np.average(storage['Energy'][n][:, select], axis=1)
                )
    return bootstrapped


storage = {'LowestEnergy': [], 'SquaredNorm': [],
           'Energy': [], 'SquaredEnergy': [], 'Nsamples': []}

setting = toml.load(open('setting.toml'))
L = setting['System']['Lattice']

for i in range(-L, L+1, 2):
    directory = './Sz='+str(i)
    AddData(directory, storage)

h_list = np.linspace(0, 4.0, 101)

nBeta = storage['beta'].size

nB = 4000
result = {
        'Beta': storage['beta'].tolist(), 'MagneticField': h_list.tolist(),
        'SystemSize': L, 'BootstrapSize': nB,
        'Energy': {'Average': [], 'Error': []},
        'Magnetization': {'Average': [], 'Error': []},
        'Susceptibility': {'Average': [], 'Error': []},
        'SpecificHeat': {'Average': [], 'Error': []},
        'SpecificHeatFromS': {'Average': [], 'Error': []},
        'Entropy': {'Average': [], 'Error': []},
        'PositivePartitionFunction': {'Average': [], 'Error': []}
          }
bootstrap_data = []
for bootstrap in range(nB):
    bootstrap_data.append(Bootstrapped(storage))

for h in h_list:
    lamb_arr = np.empty(L+1)
    for i, gene in enumerate(storage['LowestEnergy']):
        lamb_arr[i] = gene - h*(i - L/2)
    lamb_min = lamb_arr.min()
    weight = np.empty([nBeta, L+1])
    for i, beta in enumerate(storage['beta']):
        weight[i, :] = np.exp(-beta*(lamb_arr - lamb_min))

    energy = np.empty([nBeta, nB])
    entropy = np.empty([nBeta, nB])
    magnetization = np.empty([nBeta, nB])
    susceptibility = np.empty([nBeta, nB])
    specificheat = np.empty([nBeta, nB])
    specificheat_s = np.empty([nBeta, nB])
    partition = np.empty([nBeta, nB])
    for idx, data in enumerate(bootstrap_data):
        numer_M = np.sum([weight[:, i]*(i-L/2)*x
                         for i, x in enumerate(data['SquaredNorm'])],
                         axis=0)
        numer_MSq = np.sum([weight[:, i]*(i-L/2)*(i-L/2)*x
                           for i, x in enumerate(data['SquaredNorm'])],
                           axis=0)
        numer_E = np.sum([weight[:, i]*x
                         for i, x in enumerate(data['Energy'])],
                         axis=0)
        numer_EM = np.sum([weight[:, i]*(i-L/2)*x
                          for i, x in enumerate(data['Energy'])],
                          axis=0)
        numer_ESq = np.sum([weight[:, i]*x
                           for i, x in enumerate(data['SquaredEnergy'])],
                           axis=0)
        denom = np.sum([weight[:, i]*x
                       for i, x in enumerate(data['SquaredNorm'])],
                       axis=0)
        ene = numer_E / denom
        ene_sq = numer_ESq / denom
        mag = numer_M / denom
        mag_sq = numer_MSq / denom
        ene_mag = numer_EM / denom
        energy[:, idx] = ene
        entropy[:, idx] = (storage['beta']*(ene - h*mag - lamb_min)
                           + np.log(denom))
        specificheat_s[0, idx] = -storage['beta'][0] * (
                (entropy[1, idx] - entropy[0, idx])
                / (storage['beta'][1] - storage['beta'][0])
                )
        specificheat_s[1:-1, idx] = -storage['beta'][1:-1]*np.array(
                [(entropy[i+1, idx] - entropy[i-1, idx])
                 / (storage['beta'][i+1] - storage['beta'][i-1])
                 for i in range(1, nBeta-1)
                 ]
                )
        specificheat_s[-1, idx] = -storage['beta'][-1] * (
                (entropy[-1, idx] - entropy[-2, idx])
                / (storage['beta'][-1] - storage['beta'][-2])
                )
        partition[:, idx] = denom
        magnetization[:, idx] = mag
        susceptibility[:, idx] = storage['beta']*(mag_sq - np.square(mag))
        specificheat[:, idx] = np.square(storage['beta'])*(
                ene_sq - 2*h*ene_mag + h*h*mag_sq - np.square(ene - h*mag)
                )

    result['Energy']['Average'].append(np.average(energy, axis=1).tolist())
    result['Energy']['Error'].append(np.sqrt(np.var(energy, axis=1)).tolist())
    result['Entropy']['Average'].append(np.average(entropy, axis=1).tolist())
    result['Entropy']['Error'].append(
            np.sqrt(np.var(entropy, axis=1)).tolist()
            )
    result['Magnetization']['Average'].append(
            np.average(magnetization, axis=1).tolist()
            )
    result['Magnetization']['Error'].append(
            np.sqrt(np.var(magnetization, axis=1)).tolist()
            )
    result['Susceptibility']['Average'].append(
            np.average(susceptibility, axis=1).tolist()
            )
    result['Susceptibility']['Error'].append(
            np.sqrt(np.var(susceptibility, axis=1)).tolist()
            )
    result['SpecificHeat']['Average'].append(
            np.average(specificheat, axis=1).tolist()
            )
    result['SpecificHeat']['Error'].append(
            np.sqrt(np.var(specificheat, axis=1)).tolist()
            )
    result['SpecificHeatFromS']['Average'].append(
            np.average(specificheat_s, axis=1).tolist()
            )
    result['SpecificHeatFromS']['Error'].append(
            np.sqrt(np.var(specificheat_s, axis=1)).tolist()
            )
    result['PositivePartitionFunction']['Average'].append(
            np.average(partition, axis=1).tolist()
            )
    result['PositivePartitionFunction']['Error'].append(
            np.sqrt(np.var(partition, axis=1)).tolist()
            )
    print(h)

    with open('bootstrapped.json', 'w') as f:
        json.dump(result, f)
