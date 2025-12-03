#include <stdio.h>
#include <nrutils.h>
int main()
{
    float *vec = vector(0, 10);
    vec[0] = 1.0;
    printf("vec[0] = %f\n", vec[0]);
}