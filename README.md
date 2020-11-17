# DD2360_AGPUP_G13

Repository that contains Group 13 code production for assigments of course DD2360 - Applied GPU Programming @KTH

## Assignment 2

- Corresponding folder contains the code
- Bonus question is under `ex_4/` folder

Code made for and ran on Tegner, following compilation chain used (to run on K420):

```{bash}
nvcc -arch=sm_30 -I/pdc/vol/cuda/cuda-10.0/samples/common/inc exercise_X.cu -o exercise_X
```

It is also possible to generate a binary suitable for both K420 and K80 with:

```{bash}
nvcc -I/pdc/vol/cuda/cuda-10.0/samples/common/inc -gencode arch=compute_30,code=sm_30 -gencode arch=compute_37,code=sm_37 exercise_X.cu -o exercise_X
```

## Assignment 3

