# TiffCreatorRoutine
Bachelor Arbeit Projekt für das SALVE Mikroskop der Universität Ulm

# Why Python
After the acquisition of the dataset, it is practical to have an evaluation
program that can process this stack of data automatically. This can be
done in a variety of programming languages. Matlab for example as a
numerical computing environment that allows good Matrix manipulation
could be a good candidate for the task at hand. Limitations of MatLab
include a mostly inefficient use of resources and large size of the raw code.
The second obvious candidate is Python. Python is a powerful, flexible,
open source language that is easy to learn, easy to use and has powerful libraries for data manipulation and analysis. Its simple syntax is
very accessible to programming novices and will look familiar to anyone
with experience in Matlab, C/C++, Java, or Visual Basic. Python has
a unique combination of being both a capable general-purpose programming language as well as being easy to use for analytical and quantitative
computing. The Python programming language also is completely free
to use and has an excellent compatibility with virtually any operating
system. A large part of this thesis consists of writing data-preparation
functions for large image stacks and of the translation of the evaluation
program from MatLab to Python. With the complete routine, it is possible to give a single folder with raw images as input and have an output
folder where all the results are saved for future plotting. A preview of
these results is also provided. The following will describe the structure
and processes made by the program.

## Automated Image acquisition

This script will
firstly display some prompt for the user:
* Label: the user is asked if he wants to label the images. The default
value is 1 which means yes. 0 would mean no.
* Number of images: this prompt is used to set the number of images
to collect. The default value is 100.
* Binning: this prompt permits to choose the binning value if one is
desired. Possible inputs are 1,2,4 and 8.
* Exposure time: the user can choose the desired exposure time of the
chip. The default is 1 second.
* Chip Dimension: the user should enter the Chip dimension. Default
is 2048


## Program structure

Needed Python modules:
* numpy, scipy, matplotlib, math
* PIL (pillow), tifffile, libtiff, pickle
* dm3lib , tia reader, ReadDMFile


The evaluation program consists of two main classes:
* stack image v2.py: Execute this file to start the evaluation. It will
accepts 3 parameters: -srcdir [path to input folder] -output [path
to output folder] -size [NxN size of the stack (default 64x64)]. This
class will first scan the input folder and determine what kind of
images are to be processed. It will accept .ser , .tif and .dm3 formats.
For every image, it will find the maximum and cut a 64 by 64 square
around it reducing the size of the data greatly. For example 1000
images a. 16 Mb per image (a total of approx. 16 Gb) are reduced
to a single 16 Mb stack of images. This process of cutting down
is useful since the relevant information is stored at and around the
brightest point. The cropped images are now stacked so that now
one single .tif file is created where the brightest pixel is always at
the centre of every image in the stack. The cropped images are now
processed by the main program (single pixe finall all.py).

* single pixe finall all.py: Compiles and calculates all the necessary values for plotting the MTF and PTF


## Python VS Matlab

* The performance of the Python script was adequate, albeit significantly slower than the MATLAB counterpart. Since the Mathworks
have a significant cost associated with MATLAB, it is reasonable to
expect better optimisation of the provided library functions. Even so,
the Python script had adequate performance once the memory utilisation was properly optimised. Because Python is a general purpose
language, the profiling and optimisation tools are freely available and
easy to use.
* Because MATLAB uses a copy-on-write strategy, arrays can be "allocated" without actually consuming additional memory if the temporary array is unused. Since Numpy requires an explicit array copy
to be made, it is important to deallocate arrays that are no longer
needed using the del operator.
