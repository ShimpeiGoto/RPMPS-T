# RPMPS-T
This is the ITensor based implementation of random phase matrix product states with Trotter gates (RPMPS+T) approach proposed in [Phys. Rev. B 104, 045133 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.045133) ([arXiv:2103.04515](https://arxiv.org/abs/2103.04515)) to simulate quasi one-dimensional quantum many-body systems at finite temperatures.

The purpose of this implementation is to explain our approach to readers by working example.
So we dropped some optimizations especially on the composition of Trotter gates on purpose.

# Requirements
## Main C++ code (RandomMPS.cc)
This code heavily depends on ITensor library version 3.0 and later (https://github.com/ITensor/ITensor) which requires C++17.
So ITensor library should be available on your system and the code requres a compiler supporting C++17. 

For file I/O of simulations, the code uses following libraries
- JSON for Modern C++ (https://github.com/nlohmann/json)
- TOML for Modern C++ (https://github.com/ToruNiina/toml11)

We test this code with ITensor version 3.1.3, JSON for Modern C++ version 3.8.0, and TOML for Modern C++ version 3.6.0.

These are header-only libraries. Please put them (single header version of json.hpp, toml.hpp, and toml directory in these repositories) on your include path.

## Python scripts for statistical analysis and plotting
The python scripts in this repository use following non-standard libraries
- Numpy (https://pypi.org/project/numpy/)
- Matplotlib (https://pypi.org/project/matplotlib/)
- toml (https://pypi.org/project/toml/)

Please install them with your favorite tool such as pip or conda.

We test the scripts with Numpy version 1.19.4, Matplotlib version 3.3.3, and toml version 0.10.2.

# Compile
You have to compile and install ITensor library at first. After that, please copy Makefile by
```
cp Makefile.sample Makefile
```
and edit ```LIBRARY_DIR``` of the copied ```Makefile``` to point the directory where you have installed ITensor library.
Then, please type ```make``` and the compiling starts.

# How to run the main C++ program
Please run ```RandomMPS``` in a directory which contains ```setting.toml``` copied from ```setting.toml.sample```.
The output file ```sample_(random seed number).json``` will be created and updated by every iteration.

If the key "AbelianSymmetry" is set to *false*, the *grand canonical* ensemble is simulated and the key "MagneticField" is used.

If the key "AbelianSymmetry" is set to *true*, the *canonical ensemble* is simulated and the key "Sz" is used.

# How to plot with the python scripts
## In the simple canonical and grand canonical cases
One can plot thermodynamic quantities by executing the script "PlotJackknife.py" from a directory containg "setting.toml" and "sample_\*.json" files.
This script can be used for both canonical and grand canonical ensembles.

## When constructing the grand canonical ensemble from the canonical ensembles
If one would like to construct the grand canonical ensemble from the canonical ensemble in M-site system, one should perform simulations from 2Sz=-M to M and place these results into directories "TwoSz=-M" to "TwoSz=M" at first.
Next, one of "setting.toml" used in simulations and "bootstrap.toml" copied from "bootstrap.toml.sample" are placed in the same level of the directories.
In M=10 case for example, a directory structure is like this.
```bash
├── /TwoSz=-10                                                                                                                                                                    
├── /TwoSz=-8
:
:
├── /TwoSz=10
├── bootstrap.toml
└── setting.toml
```
From a directory containing "bootstrap.toml", please run the script "BootstrappedAnalysis.py". Then, "bootstrapped.json" will be generated.
By excecuting the script "PlotBootstrapped.py" from a directory with "bootstrapped.json", thermodynamic quantities are plotted.
The magnetic field to be plotted can be adjusted by modifying "bootstrap.toml".

# Create your own project
The class "randomMPS::Sampler" is designed to be compatible with any "itensor::SiteSet<>" classes such as spinful fermions or softcore bosons, and is defined in "RandomMPS.h" which depends on "RandomPhaseState.h" and the dependencies (json.hpp, toml.hpp, and itensor).
With the two header files, you can implement RPMPS+T calculations for any systems on demands.

For details, see [here](https://ShimpeiGoto.github.io/RPMPS-T/).
