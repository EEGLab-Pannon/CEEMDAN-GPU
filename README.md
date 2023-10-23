# CEEMDAN-GPU
This repo contains our CUDA implementation of the Improved CEEMDAN algorithm. The original algorithm is described in  

1. Colominas, M.A.; Schlotthauer, G.; Torres, M.E. Improved complete ensemble EMD: A suitable tool for biomedical signal processing. Biomed. Signal Process. Control 2014, 14, 19–29, doi:10.1016/j.bspc.2014.06.009.

and implemented in MATLAB available at http://perso.ens-lyon.fr/patrick.flandrin/emd.html 

Our version achieves several orders of magnitude higher speed compared to the MATLAB implementation, reducing the execution time from hours (even days) to seconds.

If you download and use our program, please cite the following paper in your relevant publications: 

Wang Z, Juhasz Z. GPU Implementation of the Improved CEEMDAN Algorithm for Fast and Efficient EEG Time–Frequency Analysis. Sensors. 2023; 23(20):8654. https://doi.org/10.3390/s23208654 

Compiling and usage instructions:

