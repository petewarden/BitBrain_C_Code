# BitBrain C Code

GitHub version of [code repository](https://figshare.manchester.ac.uk/articles/software/Implementation_of_BitBrain_algorithm_in_C/21679610)
for [BitBrain and Sparse Binary Coincidence (SBC) memories: Fast, robust learning and inference for neuromorphic architectures](https://www.frontiersin.org/articles/10.3389/fninf.2023.1125844/full)
by Michael Hopkins, Jakub Fil, Edward George Jones and Steve Furber.

Demonstrates a new technique for learning and inferring using a spiking
approach, especially suitable for constrained hardware.

## Building

If you have OpenMP and clang-8 installed, you can run this to build the main
`bb` executable:

```bash
clang-8 full_mnist_2048.c -O3 -march=native -fopenmp -lm -lomp -o bb
```

It should still be buildable with other compilers if you don't have OpenMP or
clang, with something like:

```bash
gcc full_mnist_2048.c -O3 -lm -o bb
```

Building without OpenMP will result in a slower executable, but it should be
functionally the same.

## Running

In this folder, run the executable like this from a shell:

```bash
./bb
```

On my laptop this takes about thirty seconds on a non-OpenMP build. The expected
output is:

```bash
346 wrong = 96.540 pct correct 

  970     0     2     0     0     2     4     1     1     0 
    0  1118     6     1     0     1     3     0     6     0 
    7     0  1008     2     3     0     1     6     5     0 
    1     0    11   974     0     9     0     6     7     2 
    1     0     2     0   964     0     5     0     2     8 
    5     0     3    18     1   859     4     0     2     0 
    7     2     2     0     4     5   937     0     1     0 
    1     7    25     1     4     0     0   973     5    12 
    7     0     6    13     5     9     4     5   923     2 
    8     6     7    10    26     8     1     7     8   928 

```

## License

This project is licensed under GPL v3. See LICENSE for more details.