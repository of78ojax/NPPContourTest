# Contour extraction using NPP (OpenCV for display)

Created this project since the sample project form Nvidia dodn't work for me and the description in the documentation is very poor.
The project uses NPP to extract contours from an image and OpenCV to display the results.

This is my take on trying to simplify the sample project from Nvidia and make it usable for my own projects.

## Requirements
This Project was developed on Win 11, Visual Studio 2022 and Cuda 12.6.

- NPP (Nvidia Performance Primitives) that comes with Cuda
- OpenCV (I used version 4.8.1)
- Visual Studio

### Note
I use vcpkg to install dependencies like OpenCV globally, so won't install it for every project again and again.


## How to use
1. Clone the repository
2. Open the solution in Visual Studio
3. Set / change the include paths for Cuda and OpenCv
4. Run the project

The Projects is set up to create a synthetic image with some shapes and than displays the intermediate results.

## Current state
I fidelt with the code to generate contours geometry info so far.
Next step is to extract that info and convert it to something more readable for now.