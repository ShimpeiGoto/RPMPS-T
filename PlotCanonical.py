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


def jackknife_estimate(data, func):
    nsample = data[0].size
    jackknife = np.empty(nsample)
    for i in range(nsample):
        deleted = [np.delete(x, i) for x in data]
        jackknife[i] = func(*deleted)
    jack_mean = np.average(jackknife)
    ave = func(*data)
    bias = (nsample-1)*(jack_mean - ave)
    var = np.average(np.power(jackknife - jack_mean, 2.0))
    error = np.sqrt((nsample-1)*var)
    return [ave, bias, error]


def Observable(x, y):
    return np.average(x) / np.average(y)


def Fluctuation(x, y, z):
    return np.average(x) / np.average(z) - (np.average(y) / np.average(z))**2


def FreeEnergy(x):
    return -np.log(np.average(x))


def Entropy(x, y, beta):
    return np.log(np.average(x)) + beta * np.average(y) / np.average(x)


setting = toml.load(open('setting.toml'))
N = setting['System']['Lattice']

samples = glob('sample_*.msg')

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
        beta_list = np.array(data['beta'])
    Nsample += len(data['Samples'])
    energies.append(data['LowestEnergy'])
gene = min(energies)

for i, sample in enumerate(samples):
    data = json.load(open(sample))
    for each in data['Samples']:
        norm = (np.exp(0.5*beta_list*(gene - each['Energy'][-1]))
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

sampled_ene = []
sampled_ene_err = []
sampled_Z = []
sampled_Z_err = []
sampled_C = []
sampled_C_err = []
sampled_free = []
sampled_free_err = []
sampled_entropy = []
sampled_entropy_err = []

for i, beta in enumerate(beta_list):
    Z = np.average(norm_sq_arr[i, :])
    sampled_Z.append(Z)
    sampled_Z_err.append(np.sqrt(np.var(norm_sq_arr[i, :])/Nsample))
    ave, bias, err = jackknife_estimate([ene_arr[i, :], norm_sq_arr[i, :]],
                                        Observable)
    sampled_ene.append(ave/N)
    sampled_ene_err.append(err/N)
    ave, bias, err = jackknife_estimate([ene_sq_arr[i, :],
                                         ene_arr[i, :],
                                         norm_sq_arr[i, :]],
                                        Fluctuation)
    sampled_C.append(beta*beta*ave/N)
    sampled_C_err.append(beta*beta*err/N)

    ave, bias, err = jackknife_estimate([norm_sq_arr[i, :], ene_arr[i, :]],
                                        lambda x, y: Entropy(x, y, beta))
    sampled_entropy.append((ave-beta*gene)/N)
    sampled_entropy_err.append(err/N)

    if beta > 0:
        ave, bias, err = jackknife_estimate([norm_sq_arr[i, :]], FreeEnergy)
        sampled_free.append((ave/beta + gene)/N)
        sampled_free_err.append((err/beta)/N)


plt.subplot(3, 2, 1)
plt.errorbar(beta_list, sampled_ene, yerr=sampled_ene_err)
plt.ylabel(r'$\langle \hat{H} \rangle / L$')

plt.subplot(3, 2, 2)
plt.errorbar(beta_list, sampled_Z, yerr=sampled_Z_err)
plt.axhline(y=1.0, linestyle='--', color='r')
plt.ylabel(r'$\mathrm{e}^{\beta E_0} Z(\beta)$')
plt.yscale('log')

plt.subplot(3, 2, 3)
plt.errorbar(beta_list[1:], sampled_free, yerr=sampled_free_err)
plt.ylabel(r'$-\ln Z(\beta) / (\beta L)$')

plt.subplot(3, 2, 4)
plt.errorbar(beta_list, sampled_entropy, yerr=sampled_entropy_err)
plt.axhline(y=0.0, linestyle='--', color='r')
plt.ylabel(r'$S / (k_\mathrm{B} L)$')

plt.subplot(3, 2, 5)
plt.errorbar(beta_list, sampled_C, yerr=sampled_C_err)
plt.ylabel(r'$C / (k_\mathrm{B} L)$')

plt.subplot(3, 2, 6)
plt.errorbar(beta_list, np.average(M_arr, axis=1),
             yerr=np.sqrt(np.var(M_arr, axis=1)))
plt.ylabel('Bond dimension')

plt.show()
