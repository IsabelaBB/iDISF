================================================================================
	  Interactive Dynamic and Iterative Spanning Forest (iDISF)

    Authors: Isabela B. Barcelos, Felipe C. Belém, Paulo Miranda, 
Alexandre X. Falcão, Zenilton K. G. do Patrocínio Jr. and Silvio J. F. Guimarães
================================================================================

This is an implementation of the Interactive Dynamic and Iterative Spanning Forest (iDISF). 
From an image labeled with scribbles, it performs an oversampling of background seeds and, 
through the iterations, generates superpixels and removes those considered as irrelevant. 
In this work, removal strategies by class and relevance are used. In removing by class the 
algorithm performs a desired number of iterations, and in removing by relevance the algorithm 
stops when reaching the desired amount of superpixels.

(1) Languages Supported: C/C++ (Implementation) and Python3 (Wrapper)

(2) Requirements: 

The following tools are required in order to start the installation.
	- OpenMP
	- Python 3

(2) Installing:

The following python libraries are required to compile the project.

    1. Package to build our iDISF python module: 
	pip3 install scikit-build cmake
    2. Install Tkinter libraries for python3 to run the interface: 
	sudo apt-get install python3-tk
	python3 -m pip install git+https://github.com/RedFantom/ttkthemes
    3. Install other common python libraries: 
	pip3 install -r requirements.txt
	

(3) Compiling and cleaning:

    - To compile all files: 
	make
    - For removing all generated files from source: 
	make clean

(4) Running:

In this project, there are two demo files, one for each language supported (i.e., C and Python3). 
After compiling and assuring the generation of the necessary files, one can execute each demo within its own environment. 

For a terminal located at this folder, one can run the files by the following commands:
    - C
	- The complete iDISF project can be executed by 
		./bin/DISF_demo
	- For the command options, run 
		./bin/DISF_demo --help
    - Python3
	- A demo example from iDISF in python3 (need matplotlib):
		python3 DISF_demo.py
	- It's also possible run the user interface with the follow command: 
		python3 iDISF.py

(5) Contact:
    Please, feel free to contact the authors for any unexpected behavior you might face (e.g., bugs):
	Isabela B. Barcelos: isabela.borlido@sga.pucminas.br
	Felipe C. Belém: felipe.belem@ic.unicamp.br
	Paulo Miranda: pmiranda@vision.ime.usp.br
        Alexandre X. Falcão: afalcao@ic.unicamp.br
        Silvio J. F. Guimarães: sjamil@pucminas.br

