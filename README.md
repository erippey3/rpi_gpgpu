# rpi_gpgpu
This project encompasses the different strategies for hijacking the GPU of the Raspberry Pi to perform General Purpose Computation. This includes but is not limited to testing the performance of traditional optimizations, benchmarking the VideoCore VII on Berkley Dwarfs, and testing various methods of using the GPU including clvk.


There are a total of 5 directories in this project: 

C_common - This holds helper files that are useful throughout the entire project
these include structs for matricies and vectors in the form of sparse_formats, 
device_info which helps retrieve information about the underlying hardware, 
wtime for timing, benchmark which was a struct used to hold time and more.

dwarfs - A set of benchmarks based on the Berkley Dwarfs project and originally 
forked from the work done by the OpenDwarfs project by Virginia Tech

Instruction-Support - a set of openCL kernels intended to test the capabilities of clvk running on the Raspberry Pi's VideoCore VII

proposals - a set of LaTeX and associated files that build are used to compile 
the research paper associated with this project

rpiplayground - a set of repositories based on hobbyist decompilations of 
assembly for the VideoCore IV which could proove useful in later tests 
for more efficient and performant interfacing with the VideoCore VII of the 
Raspberry Pi 5.