# Licensed under the MIT License <http://opensource.org/MIT>
#
# Copyright (c) 2020 Shimpei Goto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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


def jackknife_estimate(data, func):
    nsample = data[0].shape[1]
    jackknife = np.empty(data[0].shape)
    for i in range(nsample):
        deleted = [np.delete(x, i, axis=1) for x in data]
        jackknife[:, i] = func(*deleted)
    jack_mean = np.average(jackknife, axis=1)
    ave = func(*data)
    bias = (nsample-1)*(jack_mean - ave)
    var = np.var(jackknife, axis=1)
    error = np.sqrt((nsample-1)*var)
    return [ave, bias, error]


def Observable(x, y):
    return np.average(x, axis=1) / np.average(y, axis=1)


def Fluctuation(x, y, z):
    return (np.average(x, axis=1) / np.average(z, axis=1)
            - (np.average(y, axis=1) / np.average(z, axis=1))**2)


def FreeEnergy(x):
    return -np.log(np.average(x, axis=1))


def Entropy(x, y, beta):
    return (np.log(np.average(x, axis=1))
            + beta * np.average(y, axis=1) / np.average(x, axis=1))


setting = toml.load(open('setting.toml'))
N = setting['System']['Lattice']

samples = glob('sample_*.json')

norm_sq_arr = []
ene_arr = []
ene_sq_arr = []
M_arr = []
Nsample = 0
# Find the lowest energy

energies = []
for i, sample in enumerate(samples):
    data = json.load(open(sample))
    if i == 0:
        beta = np.array(data['beta'])
    Nsample += len(data['Samples'])
    energies.append(data['LowestEnergy'])
gene = min(energies)

for i, sample in enumerate(samples):
    data = json.load(open(sample))
    for each in data['Samples']:
        norm = (np.exp(0.5*beta*(gene - each['Energy'][-1]))
                * np.array(each['Norm']))
        ene = np.array(each['Energy'])
        ene_sq = np.array(each['SquaredEnergy'])
        norm_sq_arr.append(norm*norm)
        ene_arr.append(norm*norm*ene)
        ene_sq_arr.append(norm*norm*ene_sq)
        M_arr.append(each['BondDim'])

norm_sq_arr = np.array(norm_sq_arr).T
ene_arr = np.array(ene_arr).T
ene_sq_arr = np.array(ene_sq_arr).T
M_arr = np.array(M_arr).T

sampled_Z = np.average(norm_sq_arr, axis=1)
sampled_Z_err = np.sqrt(np.var(norm_sq_arr, axis=1)/Nsample)

ave, bias, err = jackknife_estimate([ene_arr, norm_sq_arr], Observable)
sampled_ene = ave/N
sampled_ene_err = err/N
ave, bias, err = jackknife_estimate([ene_sq_arr, ene_arr, norm_sq_arr],
                                    Fluctuation)
sampled_C = beta*beta*ave/N
sampled_C_err = beta*beta*err/N

ave, bias, err = jackknife_estimate([norm_sq_arr, ene_arr],
                                    lambda x, y: Entropy(x, y, beta))
sampled_entropy = (ave-beta*gene)/N
sampled_entropy_err = err/N

ave, bias, err = jackknife_estimate([norm_sq_arr[1:, :]], FreeEnergy)
sampled_free = (ave/beta[1:] + gene)/N
sampled_free_err = (err/beta[1:])/N


plt.subplot(3, 2, 1)
plt.errorbar(beta, sampled_ene, yerr=sampled_ene_err)
plt.ylabel(r'$\langle \hat{H} \rangle / L$')

plt.subplot(3, 2, 2)
plt.errorbar(beta, sampled_Z, yerr=sampled_Z_err)
plt.axhline(y=1.0, linestyle='--', color='r')
plt.ylabel(r'$\mathrm{e}^{\beta E_0} Z(\beta)$')
plt.yscale('log')

plt.subplot(3, 2, 3)
plt.errorbar(beta[1:], sampled_free, yerr=sampled_free_err)
plt.ylabel(r'$-\ln Z(\beta) / (\beta L)$')

plt.subplot(3, 2, 4)
plt.errorbar(beta, sampled_entropy, yerr=sampled_entropy_err)
plt.axhline(y=0.0, linestyle='--', color='r')
plt.ylabel(r'$S / (k_\mathrm{B} L)$')

plt.subplot(3, 2, 5)
plt.errorbar(beta, sampled_C, yerr=sampled_C_err)
plt.ylabel(r'$C / (k_\mathrm{B} L)$')

plt.subplot(3, 2, 6)
plt.errorbar(beta, np.average(M_arr, axis=1),
             yerr=np.sqrt(np.var(M_arr, axis=1)))
plt.ylabel('Bond dimension')

plt.show()
