# Licensed under the MIT License <http://opensource.org/MIT>
#
# Copyright (c) 2021 Shimpei Goto
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
import json

sample = json.load(open('bootstrapped.json', 'r'))

beta_sample = sample['Beta']
L = sample['SystemSize']

idx = 10
h_list = sample['MagneticField']
hz = h_list[idx]

nbeta = len(beta_sample)
plt.subplot(2, 3, 1)
plt.errorbar(beta_sample, np.array(sample['Energy']['Average'][idx])/L,
             yerr=np.array(sample['Energy']['Error'][idx])/L,
             label='bootstrap')
plt.ylabel(r'$\langle\hat{H}\rangle/L$')
plt.xlabel(r'$\beta$')
plt.legend()

plt.subplot(2, 3, 4)
plt.errorbar(beta_sample, np.array(sample['Magnetization']['Average'][idx])/L,
             yerr=np.array(sample['Magnetization']['Error'][idx])/L,
             label='bootstrap')
plt.ylabel(r'$\langle\hat{S}^z\rangle/L$')
plt.xlabel(r'$\beta$')
plt.legend()

plt.subplot(2, 3, 2)
plt.errorbar(beta_sample, np.array(sample['SpecificHeat']['Average'][idx])/L,
             yerr=np.array(sample['SpecificHeat']['Error'][idx])/L,
             label='bootstrap')
plt.errorbar(beta_sample,
             np.array(sample['SpecificHeatFromS']['Average'][idx])/L,
             yerr=np.array(sample['SpecificHeatFromS']['Error'][idx])/L,
             label='bootstrap')
plt.ylabel(r'$C/L$')
plt.xlabel(r'$\beta$')
plt.legend()

plt.subplot(2, 3, 5)
plt.errorbar(beta_sample, np.array(sample['Susceptibility']['Average'][idx])/L,
             yerr=np.array(sample['Susceptibility']['Error'][idx])/L,
             label='bootstrap')
plt.ylabel(r'$\chi/L$')
plt.xlabel(r'$\beta$')
plt.legend()

plt.subplot(2, 3, 3)
plt.errorbar(beta_sample, np.array(sample['Entropy']['Average'][idx])/L,
             yerr=np.array(sample['Entropy']['Error'][idx])/L,
             label='bootstrap')
plt.axhline(y=0.0, linestyle='--', color='r')
plt.ylabel(r'$S/L$')
plt.xlabel(r'$\beta$')
plt.legend()

plt.subplot(2, 3, 6)
plt.errorbar(beta_sample, sample['PositivePartitionFunction']['Average'][idx],
             yerr=sample['PositivePartitionFunction']['Error'][idx],
             label='bootstrap')
plt.yscale('log')
plt.ylabel(r'$\mathrm{e}^{-\beta \lambda_\mathrm{min}}\Xi$')
plt.xlabel(r'$\beta$')
plt.axhline(y=1.0, linestyle='--', color='r')
plt.legend()
plt.show()
