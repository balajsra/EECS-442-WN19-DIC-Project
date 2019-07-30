# EECS 442 DIC Project

## Useful Links
* [GitHub Repository Link](https://github.com/balajsra/eecs-442-dic-project/)

* [GitLab Repository Link](https://gitlab.eecs.umich.edu/balajsra/eecs-442-dic-project/)

* [Overleaf Report Link](https://www.overleaf.com/read/whvdsndzrvqw)

## Overview
This project investigates the application of DIC on images of a brass specimen undergoing a tensile test. These were provided by [John Laidlaw](https://me.engin.umich.edu/people/staff), Professional Engineer at the University of Michigan Mechanical Engineering Department.

The goal was to analyze displacement and strain fields in the images to create a stress-strain curve to determine material properties like Young's modulus, Poisson's ratio, ultimate tensile strength, yield strength, and fracture strength. A summary of our findings can be found in our [report](EECS_442_Computer_Vision_DIC_Project.pdf) for [EECS 442: Computer Vision](https://web.eecs.umich.edu/~fouhey/teaching/EECS442_W19/) course at the University of Michigan. You can view the latest report on [Overleaf](https://www.overleaf.com/read/whvdsndzrvqw).

Our source code for analyzing the images can be found in the **src** folder. The plots generated for the report can be found in the **Plots** folder. The images of the brass specimen can be found in the **Images**.

## Images and Data
Images of brass specimen undergoing a tensile test were provided by John Laidlaw. The TIFF images were captured with a 5 megapixel [Point Grey](https://www.flir.com/iis/machine-vision/) camera and lens. A LabView program was developed to synchronize the data and image capture throughout the tensile test. John and Dr. Thouless investigated two alternative, open-source software to perform the analysis:
* [ncorr](http://www.ncorr.com/) (MATLAB)
* [GOM Correlate](https://www.gom.com/3d-software/gom-correlate.html) (2D is free)

The data file contains a column with the frame number to sync the data and images. All the images are numbered in sequence (i.e. frame number)

## Motivation
The motivation for this project came from Sravan's [MECHENG 395: Laboratory I](https://me.engin.umich.edu/academics/courses) course at the University of Michigan in which a tensile test of brass was conducted and analyzed using [LaVision](https://www.lavision.de/en/) DaVis software. After a conversation with [Dr. Michael Thouless](https://me.engin.umich.edu/people/faculty/michael-thouless), who had worked on a paper using DIC to investigate the cohesive zone in a double-cantilever beam, we determined that this would be a cool project to combine our interest in Mechanical Engineering and Computer Science.

## Dependencies

### Recommended
* [Anaconda 5.2 for Python 3.7.x](https://www.anaconda.com/download/)

### Required
* [Numpy](https://www.numpy.org)
    * conda install -c anaconda numpy
* [Matplotlib](https://matplotlib.org)
    * conda install -c conda-forge matplotlib
* [OpenCV](https://opencv.org)
    * conda install -c menpo opencv
