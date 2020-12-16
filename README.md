# EntangledRandomPhaseMPS
# Requirements
## Main C++ code (RandomMPS.cc)
This code heavily depends on ITensor library version 3.0 and later (https://github.com/ITensor/ITensor) which requires C++17.
So ITensor library should be available on your system and the code requres a compiler supporting C++17. 

For file I/O of simulations, the code uses following libraries
- JSON for Modern C++ (https://github.com/nlohmann/json)
- TOML for Modern C++ (https://github.com/ToruNiina/toml11)

These are header-only libraries. Please put them on your include path.

## Python scripts for statistical analysis and plotting
The python scripts in this repository use following non-standard libraries
- Numpy (https://pypi.org/project/numpy/)
- Matplotlib (https://pypi.org/project/matplotlib/)
- toml (https://pypi.org/project/toml/)

Please install them with your favorite tool such as pip or conda.

# Compile
You have to compile and install ITensor library at first. After that, please copy Makefile by
```
cp Makefile.sample Makefile
```
and edit ```LIBRARY_DIR``` of the copied ```Makefile``` to point the directory where you have installed ITensor library.

# How to run
Please run ```RandomMPS``` in a directory which contains ```setting.toml``` copied from ```setting.toml.sample```.
The output file ```sample_(random seed number).json``` will be created and updated by every iteration.
