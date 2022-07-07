#### Computing Pi with Monte Carlo method

Compile it with 
```bash
nvfortran curand_m.cuf compute_pi.cuf -cudalib=curand
```
The CUDA random number generator (`curand`) needs to be linked for this to work, using the flag `-cudalib`