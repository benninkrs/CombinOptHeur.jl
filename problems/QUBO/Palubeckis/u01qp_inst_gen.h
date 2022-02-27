#include <stdio.h>

#define ALI(X,Z) if ((X=(int *)calloc(Z,sizeof(int)))==NULL) \
       {fprintf(out,"  failure in memory allocation\n");exit(0);}
#define ALM(X,Z) if ((X=(int **)calloc(Z,sizeof(int *)))==NULL) \
       {fprintf(out,"  failure in memory allocation\n");exit(0);}

#define c(X,Y) *(*(pmatrix+X)+Y)
