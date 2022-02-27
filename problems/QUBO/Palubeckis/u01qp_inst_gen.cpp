/* Program: generator of instances of the unconstrained binary 
     quadratic optimization problem
   Author: Gintaras Palubeckis
   Date: 2002-03-21
   Language: C
   The program produces the matrix of the coefficients of the objective 
   function. Its density is specified by a parameter. All nonzero 
   entries of the matrix are drawn uniformly from the specified 
   interval. The generated numbers are written down to an output file.
   Some of the input data are supplied through parameters and the rest 
   through the (input) file.
   Parameters:
     - input file name;
     - output file name.
   An example of invocation from the command line:
     u01qp_inst_gen.exe in5000_1.txt D:\gener\p5000_1.dat;
   Input file contains:
     - number of variables in an instance to be generated;
     - density of the coefficients matrix;
     - left end of the interval from which coefficients of the linear 
       part of the objective function will be drawn;
     - right end of the interval from which coefficients of the linear 
       part of the objective function will be drawn;
     - left end of the interval from which coefficients of the quadratic 
       part of the objective function will be drawn;
     - right end of the interval from which coefficients of the quadratic 
       part of the objective function will be drawn;
     - seed for random number generator;
   Example of the input file:
   5000 50 -100 100 -100 100 51000
*/



//#include <alloc.h>
//#include <process.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "u01qp_inst_gen.h"


 double random(double *seed,double coef)
 {double rd,rf;
  rd=16807*(*seed);rf=floor(rd/coef);
  *seed=rd-rf*coef;
  return(*seed/(coef+1));
 }


 int main(int argc,char **argv)
 {FILE *out,*in;
  char in_file_name[80];
  char out_file_name[80];
  double seed,coef;
  int **pmatrix;

  int i,i1,j,j1,k;
  int r;
  int v_count;
  int density;
  int lb_linear,ub_linear,lb_quadr,ub_quadr;
  float fl;

  if (argc<=2) {printf("  specify input and instance files");exit(1);}
  strcpy(in_file_name,argv[1]);
  strcpy(out_file_name,argv[2]);

  coef=2048;coef*=1024;coef*=1024;coef-=1;
  if ((in=fopen(in_file_name,"r"))==NULL)
     {printf("  fopen failed for input");exit(1);}
  fscanf(in,"%d %d %d %d %d %d %lf",&v_count,&density,&lb_linear,&ub_linear,
      &lb_quadr,&ub_quadr,&seed);
  if ((out=fopen(out_file_name,"w"))==NULL)
     {printf("  fopen failed for output  %s",out_file_name);exit(1);}
  ALM(pmatrix,v_count+1)
  for (i=0;i<=v_count;i++) ALI(*(pmatrix+i),v_count+1)
  for (i=1;i<=v_count;i++)
     {r=random(&seed,coef)*(ub_linear-lb_linear+1);c(i,i)=r+lb_linear;
      for (j=i+1;j<=v_count;j++)
	      {fl=random(&seed,coef)*100;
	       if (fl<=density)
	          {r=random(&seed,coef)*(ub_quadr-lb_quadr+1);c(i,j)=r+lb_quadr;}
	        else c(i,j)=0;
	       c(j,i)=c(i,j);
	      }
     }
  fprintf(out,"%3d %7d\n",1,v_count);
  k=v_count/15;if (v_count%15!=0) k++;
  for (i=1;i<=v_count;i++) 
     {j1=0;
      for (i1=1;i1<=k;i1++)
         {for (j=1;j<=15;j++)
             {j1++;if (j1>v_count) break;
              fprintf(out,"%5d",c(i,j1));
             }
          fprintf(out,"\n");
         }
     }
  for (i=1;i<=v_count;i++) if (*(pmatrix+i)!=NULL) free(*(pmatrix+i));
  if (pmatrix!=NULL) free(pmatrix);
  fclose(out);
  fclose(in);
 }

