#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

typedef struct {
	double n0;
	double n1;
	double n2;
}Norms;

typedef struct {
	double comm;
	double time_step;
	double total;
}Times;

const int FINAL_TIME = 5;
int GRID_SIZE;
const double INITIAL_POINT = -50.0;
const double FINAL_POINT = 50.0;
const double U = 1.75;
const double CFL = 0.75;
double deltaX;
double deltaT;
double *numerical;
double *numerical_solution;
double *analytical;
double *analytical_solution;
Norms norms;
Norms norms_solution;
int npes, myrank;
MPI_Status status;
int local_gridsize;
int index_start, index_end, extra;
Times times;
Times times_solution;

double *double_dup(double const *src, int len)
{
	double *p = malloc(len * sizeof(double));
	memcpy(p, src, len * sizeof(double));
	return p;
}


double calculate_analytical(double x, double t)
{
	return exp(-(pow((x - 1.75 * t), 2))) / 2;
}

void update_norms()
{
	double max = 0.0, sum1 = 0.0, sum2 = 0.0, error;
	int i;
	for (i = 0; i < index_end; i++)
	{
		error = numerical[i] - analytical[i];
		max = fabs(error) > max ? fabs(error) : max;
		sum1 += fabs(error);
		sum2 += pow(error, 2);
	}
	norms.n0 = max;
	norms.n1 = sum1;
	norms.n2 = sum2;
}

void initialize()
{
	index_start = 0;
	if(extra)
	{
		index_end = local_gridsize - 1;
		numerical[index_end] = NAN;
	}
	else
		index_end = local_gridsize;
		
	if(myrank == 0)
	{
		numerical[0] = 0;
		index_start++;
		analytical[0] = calculate_analytical(INITIAL_POINT, 0);
	}
	if(myrank == npes - 1)
	{
		numerical[index_end - 1] = 0;
		analytical[index_end - 1] = calculate_analytical(FINAL_POINT, 0);
		index_end--;
	}	
		


	int i;
	double realX;
	for (i = index_start; i < index_end; i++)
	{
		realX = INITIAL_POINT + (i + myrank * local_gridsize) * deltaX;
		numerical[i] = exp(-(pow(realX,2))) / 2;
		analytical[i] = calculate_analytical(realX, 0);
	}

	times.comm = 0;
	times_solution.comm = 0;
	times.time_step = 0;
	times_solution.time_step = 0;
}

void writeToFile(char *fileName)
{
	char realFileName[20];
	sprintf(realFileName,"%s_%d_%d.csv",fileName,GRID_SIZE, npes);

	FILE *file;
	file = fopen(realFileName, "w");
	if (file == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	int i, real_index;
	fprintf(file, "norm0,norm1,norm2,communication,timestep,total,position,numerical,analitical\n");
	real_index = 0;
	int bigSize = local_gridsize * npes;
	fprintf(file, "%f,%f,%f,%f,%f,%f,", norms_solution.n0, norms_solution.n1, norms_solution.n2, times_solution.comm, times_solution.time_step, times_solution.total);
	for (i = 0; i < bigSize; i++)
	{
			
		if(numerical_solution[i] != numerical_solution[i])
			i++;
		if(i < bigSize)
			fprintf(file, "%g,%g,%g\n",INITIAL_POINT + real_index * deltaX, numerical_solution[i], analytical_solution[i]);
		fprintf(file," , , , , , ,");
		real_index++;
	}

	fclose(file);
}

void gatherSolution()
{
	MPI_Gather(numerical, local_gridsize, MPI_DOUBLE, numerical_solution, local_gridsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(analytical, local_gridsize, MPI_DOUBLE, analytical_solution, local_gridsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	MPI_Reduce(&norms.n0,&norms_solution.n0,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	MPI_Reduce(&norms.n1,&norms_solution.n1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&norms.n2,&norms_solution.n2,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(myrank == 0)
		norms_solution.n2 = sqrt(norms_solution.n2);
	
}

void gatherTimes()
{
    MPI_Reduce(&times.comm ,&times_solution.comm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&times.time_step,&times_solution.time_step,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(myrank == 0)
	{
		times_solution.comm = times_solution.comm / ((double)npes);
		times_solution.time_step = times_solution.time_step / ((double)npes);
	}
}

void explicit_upwind()
{
	double total1,total2,comm1,comm2,time_s1,time_s2;
	if(myrank == 0)
		total1 = MPI_Wtime();
	initialize();
	double ratio = U * (deltaT / deltaX);
	double realX;
	double temp;
	double firstValue;
	
	double t;

	int x;
	//iterate for each time step
	for (t = deltaT; t <= FINAL_TIME; t += deltaT)
	{
		time_s1 = MPI_Wtime();

		if(myrank == 0)
			analytical[0] = calculate_analytical(INITIAL_POINT, t);
		if(myrank == npes - 1)
			analytical[index_end - 1] = calculate_analytical(FINAL_POINT, t);

		//wait for send/receive of last/first value in space
		if(myrank != 0)
		{
			
			comm1 = MPI_Wtime();
			MPI_Recv(&firstValue,1,MPI_DOUBLE,myrank-1,t,MPI_COMM_WORLD,&status);
			comm2 = MPI_Wtime();
			times.comm += comm2 - comm1;
		}
		else
			firstValue = numerical[0];
		//iterate for each point in space
		for (x = index_start; x < index_end; x++)
		{
			temp = numerical[x];
			numerical[x] = numerical[x] - ratio * (numerical[x] - firstValue);
			firstValue = temp;

			realX = INITIAL_POINT + (x + myrank * local_gridsize) * deltaX;
			analytical[x] = calculate_analytical(realX, t);
		}
		if(myrank != (npes - 1))
		{
			comm1 = MPI_Wtime();
			MPI_Send(&firstValue,1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD);
			comm2 = MPI_Wtime();
			times.comm += comm2 - comm1;
		}

		time_s2 = MPI_Wtime();
		times.time_step += time_s2 - time_s1;
	}
	double test1,test2,test3;
	test1 = MPI_Wtime();
    update_norms();

	gatherSolution();
	test2 = MPI_Wtime();
	test3 = test2-test1;
	

	if(myrank == 0)
	{
	  	total2 = MPI_Wtime();
		times.total = total2 - total1;
		times_solution.total = times.total;
		printf("That took %f and results in %f versus %f\n",test3,test3+times.time_step, times.total);
	}
    int iterations = (FINAL_TIME / deltaT);
    times.comm = times.comm / ((double)iterations);
    times.time_step = times.time_step / ((double)iterations);
    gatherTimes();

	if(myrank == 0){
		writeToFile("ExplicitUpwind");
	}
}

double buildCPrime(double *cPrime,double a,double b,double c,double t)
{
	double comm1,comm2, lastValue = 0;
	if(myrank == 0)
		cPrime[0] = c / b;
	else
	{
		comm1 = MPI_Wtime();
		MPI_Recv(&lastValue,1,MPI_DOUBLE,myrank-1,t,MPI_COMM_WORLD,&status);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
		cPrime[0] = c / (b - a * lastValue);
	}
	int i;
	for(i = 1; i < index_end; i++) //the last processor will not use the last value of cPrime
	{
		cPrime[i] = c / (b - a * cPrime[i - 1]);
	}

	if(myrank != npes - 1)
	{
		comm1 = MPI_Wtime();
		MPI_Send(&cPrime[index_end-1],1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
	}

	return lastValue;
}

void buildVector(double *vec, double beta, double t)
{
	double comm1, comm2;
	int i;
	double lastValue,firstValue;
	if(myrank != (npes - 1))
	{
		comm1 = MPI_Wtime();
		MPI_Sendrecv(&numerical[index_end - 1],1,MPI_DOUBLE,myrank+1,t+10,
			&lastValue,1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD,&status);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
	}
	if(myrank != 0)
	{
		comm1 = MPI_Wtime();
		MPI_Sendrecv(&numerical[0],1,MPI_DOUBLE,myrank-1,t,
			&firstValue,1,MPI_DOUBLE,myrank-1,t+10,MPI_COMM_WORLD,&status);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
	}
	if(myrank == 0)
		vec[0] = numerical[0] - beta * numerical[1];
	else
		vec[0] = beta * firstValue + numerical[0] - beta * numerical[1];
	
	for(i = 1; i < index_end - 1; i++)
	{
		vec[i] = beta * numerical[i-1] + numerical[i] - beta * numerical[i+1];
	}
	if(myrank == (npes - 1))
		vec[index_end - 1] = numerical[index_end - 1] + beta * numerical[index_end - 2];
	else
		vec[index_end - 1] = beta * numerical[index_end-2] + numerical[index_end - 1] - beta * lastValue;
}

void triDiagonalSolve(double a, double b, double c, double *vec, double t, double *cPrime, double lastCPrime)
{
	double comm1, comm2;
	
	double dPrime[index_end];
	
	if(myrank == 0)
		dPrime[0] = vec[0] / b;
	else
	{
		double lastValue;
		comm1 = MPI_Wtime();
		MPI_Recv(&lastValue,1,MPI_DOUBLE,myrank-1,t,MPI_COMM_WORLD,&status);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
		dPrime[0] = (vec[0] - a * lastValue) / (b - a * lastCPrime);
	}

	int i;
	for(i = 1; i < index_end; i++)
	{
		dPrime[i] = (vec[i] - a * dPrime[i - 1]) / (b - a * cPrime[i - 1]);
	}

	if(myrank != npes - 1)
	{
		comm1 = MPI_Wtime();
		MPI_Send(&dPrime[index_end-1],1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
	}

	if(myrank == npes - 1)
	{
		numerical[index_end - 1] = dPrime[index_end - 1];
	}
	else
	{
		double lastValue;
		comm1 = MPI_Wtime();
		MPI_Recv(&lastValue,1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD,&status);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
		numerical[index_end - 1] = dPrime[index_end-1] - cPrime[index_end-1] * lastValue;
	}

	for(i = index_end - 2; i >= 0; i--)
		numerical[i] = dPrime[i] - cPrime[i] * numerical[i + 1];

	if(myrank != 0)
	{
		comm1 = MPI_Wtime();
		MPI_Send(&numerical[0],1,MPI_DOUBLE,myrank-1,t,MPI_COMM_WORLD);
		comm2 = MPI_Wtime();
		times.comm += comm2 - comm1;
	}
	
}

void crank_nicolson()
{
	double total1,total2,time_s1,time_s2;
	if(myrank == 0)
		total1 = MPI_Wtime();
	initialize();
	double realX;
	double *temp;
	temp = (double*)malloc((index_end) * sizeof(double));
	//Coefficents of the n+1 matrix. Since this will be a tridiagonal matrix with the same values throughout it,
	//we only need to store the 3 different values
	double a,b,c;
	double beta = U * deltaT / (4 * deltaX);
	a = -beta;
	b = 1;
	c = beta;
	double cPrime[index_end - 1];
	time_s1 = MPI_Wtime();
	double lastCPrime = buildCPrime(cPrime,a,b,c,0);
	double t;
	int x;
	for (t = deltaT; t <= FINAL_TIME; t += deltaT)
	{
		if(t != deltaT)
			time_s1 = MPI_Wtime();
		buildVector(temp, beta, t);
		triDiagonalSolve(a,b,c,temp, t, cPrime, lastCPrime);
		for (x = index_start; x < index_end; x++)
		{
			realX = INITIAL_POINT + (x + myrank * local_gridsize) * deltaX;
			analytical[x] = calculate_analytical(realX, t);
		}
		time_s2 = MPI_Wtime();
		times.time_step += time_s2 - time_s1;
	}
	double test1,test2,test3;
	test1 = MPI_Wtime();
    update_norms();

	gatherSolution();
	test2 = MPI_Wtime();
	test3 = test2-test1;
	if(myrank == 0)
	{
	  	total2 = MPI_Wtime();
		times.total = total2 - total1;
		times_solution.total = times.total;
		printf("That took %f and results in %f versus %f\n",test3,test3+times.time_step, times.total);
	}
    int iterations = (FINAL_TIME / deltaT);
    times.comm = times.comm / ((double)iterations);
    times.time_step = times.time_step / ((double)iterations);
    gatherTimes();

	if(myrank == 0)
	{
		writeToFile("CrankNicolson");
	}
}
void implicit_upwind()
{
	double total1,total2,comm1,comm2,time_s1,time_s2;
	if(myrank == 0)
		total1 = MPI_Wtime();
	initialize();
	double ratio = U * (deltaT / deltaX);
	double realX;
	double firstValue;
	
	double t;

	int x;
	//iterate for each time step
	for (t = deltaT; t <= FINAL_TIME; t += deltaT)
	{
		time_s1 = MPI_Wtime();
		if(myrank == 0)
		{	
			numerical[0] = numerical[0] / (1+ratio);
			analytical[0] = calculate_analytical(INITIAL_POINT, t);
		}
		if(myrank == npes - 1)
			analytical[index_end - 1] = calculate_analytical(FINAL_POINT, t);

		//wait for send/receive of last/first value in space
		if(myrank != 0)
		{
			comm1 = MPI_Wtime();
			MPI_Recv(&firstValue,1,MPI_DOUBLE,myrank-1,t,MPI_COMM_WORLD,&status);
			comm2 = MPI_Wtime();
			times.comm += comm2 - comm1;
		}
		else
			firstValue = numerical[0];
		//iterate for each point in space
		for (x = index_start; x < index_end; x++)
		{
			numerical[x] = (numerical[x] + firstValue * ratio) / (1.0 + ratio);
			firstValue = numerical[x];

			realX = INITIAL_POINT + (x + myrank * local_gridsize) * deltaX;
			analytical[x] = calculate_analytical(realX, t);
		}
		if(myrank != (npes - 1))
		{
			comm1 = MPI_Wtime();
			MPI_Send(&numerical[index_end - 1],1,MPI_DOUBLE,myrank+1,t,MPI_COMM_WORLD);
			comm2 = MPI_Wtime();
			times.comm += comm2 - comm1;
		}

		time_s2 = MPI_Wtime();
		times.time_step += time_s2 - time_s1;
	}
 	double test1,test2,test3;
	test1 = MPI_Wtime();
    update_norms();

	gatherSolution();
	test2 = MPI_Wtime();
	test3 = test2-test1;

	if(myrank == 0)
	{
	  	total2 = MPI_Wtime();
		times.total = total2 - total1;
		times_solution.total = times.total;
		printf("That took %f and results in %f versus %f\n",test3,test3+times.time_step, times.total);
	}
    int iterations = (FINAL_TIME / deltaT);
    times.comm = times.comm / ((double)iterations);
    times.time_step = times.time_step / ((double)iterations);
    gatherTimes();
	
	if(myrank == 0){
		writeToFile("ImplicitUpwind");
	}
}
void lapack_implicit()
{
	int size = 3;
	int i, j , c1, c2, pivot[size], ok;
	float A[size][size], b[size], AT[size*size];	/* single precision!!! */


	A[0][0]=3.1;  A[0][1]=1.3;  A[0][2]=-5.7;	/* matrix A */
	A[1][0]=1.0;  A[1][1]=-6.9; A[1][2]=5.8;	
	A[2][0]=3.4;  A[2][1]=7.2;  A[2][2]=-8.8;	

	b[0]=-1.3;			/* if you define b as a matrix then you */
	b[1]=-0.1;			/* can solve multiple equations with */
	b[2]=1.8;			/* the same A but different b */ 	
	 
	for (i=0; i<size; i++)		/* to call a Fortran routine from C we */
	{				/* have to transform the matrix */
	  for(j=0; j<size; j++) AT[j+size*i]=A[j][i];		
	}						

	c1=size;			/* and put all numbers we want to pass */
	c2=1;    			/* to the routine in variables */

	/* find solution using LAPACK routine SGESV, all the arguments have to */
	/* be pointers and you have to add an underscore to the routine name */
	LAPACKE_sgesv(&c1, &c2, AT, &c1, pivot, b, &c1, &ok);      

	/*
	 parameters in the order as they appear in the function call
	    order of matrix A, number of right hand sides (b), matrix A,
	    leading dimension of A, array that records pivoting, 
	    result vector b on entry, x on exit, leading dimension of b
	    return value */ 
	     						
	for (j=0; j<size; j++) printf("%e\n", b[j]);	/* print vector x */
}

void compute(int grid_size)
{
	GRID_SIZE = grid_size;
	deltaX = (FINAL_POINT - INITIAL_POINT) / (GRID_SIZE - 1.0);
	deltaT = (deltaX * CFL) / U;

	int leftover = GRID_SIZE % npes;
	local_gridsize = (GRID_SIZE / npes);
	if(leftover > 0)
	{
		local_gridsize++;
		if(myrank < leftover)
			extra = 0;
		else
			extra = 1;
	}
	else
	{
		extra = 0;
	}

	numerical = (double*)malloc(local_gridsize * sizeof(double));
	analytical = (double*)malloc(local_gridsize * sizeof(double));
	if(myrank == 0)
	{
		numerical_solution = (double*)malloc((local_gridsize * npes) * sizeof(double));
		analytical_solution = (double*)malloc((local_gridsize * npes) * sizeof(double));
	}


	explicit_upwind();
	//dark magic
	explicit_upwind();
	crank_nicolson();
	implicit_upwind();

	free(numerical);
	free(analytical);
	if(myrank == 0){
		free(numerical_solution);
		free(analytical_solution);
	}
}


int main(int argc, char *argv[])
{
	lapack_implicit();
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	//compute(1000);
	

	//compute(10000);

	MPI_Finalize();
	return 0;
}
