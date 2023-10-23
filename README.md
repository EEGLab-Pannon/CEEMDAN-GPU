# CEEMDAN-GPU
This repo contains our CUDA implementation of the Improved CEEMDAN algorithm. The original algorithm is described in  

- Colominas, M.A.; Schlotthauer, G.; Torres, M.E. Improved complete ensemble EMD: A suitable tool for biomedical signal processing. Biomed. Signal Process. Control 2014, 14, 19–29, doi:10.1016/j.bspc.2014.06.009.

and implemented in MATLAB available at http://perso.ens-lyon.fr/patrick.flandrin/emd.html 

Our version achieves several orders of magnitude higher speed compared to the MATLAB implementation, reducing the execution time from hours (even days) to seconds.

If you download and use our program, please cite the following paper in your relevant publications: 

- Wang Z, Juhasz Z. GPU Implementation of the Improved CEEMDAN Algorithm for Fast and Efficient EEG Time–Frequency Analysis. Sensors. 2023; 23(20):8654. https://doi.org/10.3390/s23208654 

## Compiling and usage instructions

The supplied code has been tested in Linux and Windows environments. On Linux, compile the code as follows after changing the -arch parameter to match your target CUDA device architecture. This example assumes Volta architecture, compute capability 7.0. 

The implementation can be found in the source files cudaICEEMDAN.cu and statistics.cu. We have provided two sample C program files that demonstrate how to use the implementation. The program sample_synthetic_signal.cu creates and uses synthetic signals, while the file sample_binary_file.cu demonstrates how to read in a binary input data file generated in, say, MATLAB and process on the GPU.      

`nvcc -arch=sm_70 -Xcompiler -fopenmp -lcublas -lcusparse -lcurand ./sample_synthetic_signal.cu ./cudaICEEMDAN.cu ./statistics.cu -o CUDA_ICEEMDAN `

On Windows systems, we recommend using the Visual Studio or Visual Studio Code environment. Create a CUDA project, place the file from this project into the source directory and add them to the VS project. After setting the necessary project configuration parameters, the project can be built. 

We have also providing an implementation for direct execution from MATLAB. This can be downloaded from the [folder](./MatlabMEXCUDA).
