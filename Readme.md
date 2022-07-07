### Fortran programming on GPU : CUDA Fortran

#### Installation:
1. Install the appropriate Nvidia drivers for your system. 
2. Install the Nvidia CUDA toolkit. 
3. Install Nvidia HPC SDK from https://developer.nvidia.com/nvidia-hpc-sdk-downloads. The installation path is usually `/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin`, add it to your PATH. 

P.S. You may have to restart your system, before using the compilers.

### Compilation and Execution

Earlier the CUDA Fortran compiler was developed by PGI. From 2020 the PGI compiler tools was replaced with the Nvidia HPC Toolkit. You can use compilers like `nvc`, `nvc++` and `nvfortan` to compile `C`, `C++` and `Fortran` respectively.

- CUDA Fortran codes have suffixed `.cuf`

- Compile CUDA Fortran with `nvfortran` and just run the executable

```bash
nvfortran test_code.cuf -o test_exe
./test_exe
```

- Profile the code with `nvprof`, this can tell about the time spent on each GPU activities
```bash
nvprof ./test_exe
```





### CUDA Fortran Code:

- Disclaimer: There is no way possible to learn CUDA Fortran completely just from this one page Tutorial/Cheatsheet. This is only meant for a quick reference sheet to get started with GPGPU programming with CUDA Fortran



- Will follow the SAXPY (Scalar A*X Plus Y) problem to show how to go from CPU to GPU code. 

The serial CPU code
```Fortran
module mathOps
    contains
       subroutine saxpy(x, y, a)
        implicit none
        real :: x(:), y(:), a
        ! Just a simple array scaler multiplication and addition
        y = a*x +y
      end subroutine saxpy 
    end module mathOps
    
    program testSaxpy
      use mathOps
      implicit none
      integer, parameter :: N = 40000
      real :: x(N), y(N), a

      x = 1.0; y = 2.0; a = 2.0

      write(*,*) 'Max error: ', maxval(abs(y-4.0))
    end program testSaxpy 
````

Now the GPU code:

```fortran
module mathOps
  contains
  ! The kernel i.e a function that runs on the device 
  ! `attributes` describes the scope of the routine. `global` means its visible both from the host and device
  ! This indicates the subroutine is run on the device but called from the host
    attributes(global) subroutine saxpy(x, y, a)
      implicit none
      real :: x(:), y(:)
      real, value :: a
      integer :: i, n
      n = size(x)

      ! Remember the host launches "**grid** of block with each block having **tBlock** threads"
      ! and each thread works on a single element of the array
      ! These `blockDim`, `blockIdx` and `threadIdx` are provided defined by CUDA are similar to `dim3` type
      ! As we used only `x` component to launch the kernel only `x` component is used

      ! Think of this as there are groups (=`grid` in host) of threads and those groups are numbered with `blockIdx`
      ! Each group has `blockDim` (=`tBlock` in host) number of threads and each thread inside a particular block
      ! is numbered with `threadIdx`. 
      ! Thus using this following formula we can calculate the offset of the element of array to be computed

      i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
      !^ Note: This is an example of fine-grained parallelism

      ! As we have launched more threads in the host than there are array element a conditional check is required
      if (i <= n) y(i) = y(i) + a*x(i)

    end subroutine saxpy 
  end module mathOps
  
  program testSaxpy
    use mathOps
    ! Fortran module that contains all the CUDA Fortran definitions
    use cudafor
    implicit none
    integer, parameter :: N = 40000
    ! host arrays
    real :: x(N), y(N), a
    ! device arrays, they are declared with the `device` attribute
    real, device :: x_d(N), y_d(N)
  
    ! Thread configuration to launch the kernel
    ! Threads and block can be arranged in multidimensional nature. 
    ! Here the they are defined as `dim3`, so they as have three components `x`,`y` and `z`.
    type(dim3) :: grid, tBlock
  
    
    ! In this example `tBlock` has 256 threads in `x` components and 
    ! `grid` is defined such a way to accomodate all the `N = 40000` computation in blocks each having 256 threads.
    tBlock = dim3(256,1,1)
    grid = dim3(ceiling(real(N)/tBlock%x),1,1)
  
    x = 1.0; y = 2.0; a = 2.0
  
    ! copies the array from host to device and vice versa. 
    ! The `cudafor` module overloads the assignment operator with `cudaMemcpy` calls 
    ! so that the memory transfer can be done with a simple assignment operation. 
    ! Note: This step actually moves data beween two physical devices and actually can be time consuming.
    ! P.S. CUDA memory copy can also be done in asynchronous
    x_d = x
    y_d = y

    ! Launches the kernel on the device. 
    ! The information between the tripple braces are the excution configuration 
    ! it says to launch **grid** of block with each block having **tBlock** threads.
    ! This call is asynchronous, so the host can just call this and 
    ! proceed to the next line without it's computation on the device being completed
    call saxpy<<<grid, tBlock>>>(x_d, y_d, a)


    ! Copyies data back to host. 
    ! This process is synchronous and waits for the device to complete the calculation
    y = y_d
    write(*,*) 'Max error: ', maxval(abs(y-4.0))
  end program testSaxpy
```








#### Terminologies:
1. host & device: 
  The CPU and the GPU respectively. Each have their own cores and memory. The user does not have direct access to the GPU and only has access to the host the CPU aka the host. User starts the computation from a host. To do computation on the device the data has to be on the device memory and user has t



nvprof

cuda memcpy can break, async

loop parallel
loop parallel multiple























### References
1. __CUDA Fortran for Scientists and Engineers__ by Gregory Ruetsch & Massimiliano Fatica
2. https://developer.nvidia.com/blog/easy-introduction-cuda-fortran/