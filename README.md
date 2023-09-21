# Interactive Dynamic and Iterative Spanning Forest (iDISF)

![](https://img.shields.io/apm/l/vim-mode) ![](https://img.shields.io/pypi/implementation/C) ![](https://img.shields.io/pypi/pyversions/tk) 

This repository contains a tool for interactive image segmentation and an iDISF implementation.

### Interactive DISF (iDISF)

From an image labeled with scribbles, it performs an oversampling of background seeds and, 
through the iterations, generates superpixels and removes those considered as irrelevant. 
In this work, removal strategies by class and relevance are used. In removing by class the 
algorithm performs a desired number of iterations, and in removing by relevance the algorithm 
stops when reaching the desired amount of superpixels.

### Languages supported

C/C++ (Implementation) and Python3 (Wrapper)

## Requirements

The following tools are required in order to start the installation.
- OpenMP
- Python 3

## Compiling and cleaning

- To compile all files: `make`
- For removing all generated files from source: `make clean`

## Installing the python interface

1. Install Tkinter library for python3 to run the interface: `sudo apt-get install python3-tk`

2. To use the graphical user interface tool, we recommend to install the required python packages using pipenv:
  
    - Install pipenv: `pip3 install pipenv`

    - Create a virtual environment and install the required packages from PipFile: `pipenv install`

    - Install iDISF package (make sure you successfully run `make` before it): `cd python3/; pipenv run python3 -m pip install . ; cd ..`

The following python libraries are required to run the graphical user interface.
- Pillow
- Numpy
- Scipy
- opencv-python
- opencv-contrib-python
- Higra
- Matplotlib
- Pydicom
- ttkthemes

For those not using any virtual environment, the iDISF package must be installed with: `cd python3/; python3 -m pip install . ; cd ..`

If an error occurs due to a file not found when installing the iDISF package, consider changing directories in `python3/setup.py`.

## Running

In this project, there are two demo files, one for each language supported (i.e., C and Python3). After compiling and assuring the generation of the necessary files, one can execute each demo within its own environment. 

### iDISF only

- The complete iDISF project can be executed by `./bin/DISF_demo`
- For the command options, run `./bin/DISF_demo --help`
- A demo example from iDISF in python3: `python3 DISF_demo.py`

### Interface

- For those using pipenv, run the user interface with the follow command: `pipenv run python3 iDISF.py`
- For those not using any virtual environment, run the user interface with the follow command: `python3 iDISF.py`
- Our interface includes iDISF and Watershed segmentation. For Watershed, we use [higra](https://github.com/higra/Higra) package.

## Cite
If this work was useful for your research, please cite our paper:

```
@InProceedings{barcelos2021towards,
  title={Towards Interactive Image Segmentation by Dynamic and Iterative Spanning Forest},
  author={Barcelos, Isabela Borlido and Bel{\'e}m, Felipe and Miranda, Paulo and Falc{\~a}o, Alexandre Xavier and do Patroc{\'\i}nio, Zenilton KG and Guimar{\~a}es, Silvio Jamil F},
  booktitle={International Conference on Discrete Geometry and Mathematical Morphology},
  publisher={Springer International Publishing},
  pages={351--364},
  year={2021},
  organization={Springer}
}
```

## Contact

Please, feel free to contact the authors for any unexpected behavior you might face (e.g., bugs). Moreover, if you’re interested in using our code, we appreciate if you cite this work in your project.

- Isabela Borlido: isabela_borlido@hotmail.com
- Felipe C. Belém: felipe.belem@ic.unicamp.br
- Zenilton K. G. Do Patrocínio Jr.  zenilton@pucminas.br
- Alexandre X. Falcão: afalcao@ic.unicamp.br
- Silvio J. F. Guimarães: sjamil@pucminas.br
