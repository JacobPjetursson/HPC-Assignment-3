#define FLOP 5


__global__ void jacobiIteration(double *u, double *uold, double *f, int N);
__global__ void jacobiIteration_per_elem(double *u, double *uold, double *f, int N);
__global__ void jacobiIteration_per_elem_2(double *u, double *uold, double *uold_2, double *f, int N, int device);

#ifndef C_ITERATOR_H
#define C_ITERATOR_H

#endif //C_ITERATOR_H
