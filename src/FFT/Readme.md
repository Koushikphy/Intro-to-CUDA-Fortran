#### Fast Fourier Transform with CUDA Fortran

Compile it with 
```bash
nvfortran cufft_m.cuf fft_test_c2c.cuf -cudalib=cufft
nvfortran cufft_m.cuf fft_derivative.cuf -cudalib=cufft
```
The CUDA random number generator (`curand`) needs to be linked for this to work, using the flag `-cudalib`
