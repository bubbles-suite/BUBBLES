#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>
#include "mdgpu_np.cuh"

int main ( int argc, char * argv[] ) 
{
	/// initialize simulation
	initialize(argc, argv);
	
	if(verbose)
		printf("Starting simulation with %d particles\n", N);
	
	/** Create the file that contains all the coordinates with all the frames of the movie */
	FILE * movie;
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s%d%s", root, "results/movie/movie_run", restartIndex, ".xyz");
	snprintf(filename, sizeof(filename), "%s%s", root, "results/movie/movie.xyz");
	if(restart) movie = fopen(filename,"a");    /// open in append mode
	else        movie = fopen(filename,"w");    /// open in overwrite mode
	
	if (movie == NULL) {
		printf("Error: file movie.xyz could not be opened.\n");
		exit(1);
	}
	
	/** Create the file that contains all the coordinates of  all the frames of the movie during adsorption in Logscale */
	FILE * movieLog;
	snprintf(filename, sizeof(filename), "%s%s", root, "results/movie/movieLog.xyz");
	if(restart) movieLog = fopen(filename,"a");    /// open in append mode
	else        movieLog = fopen(filename,"w");    /// open in overwrite mode
	if (movieLog == NULL) {
		printf("Error: file movieLog.xyz could not be opened.\n");
		exit(1);
	}
	
	/** Create the file that contains the adsorption profiles in linear time */
	FILE * adsorption;
	snprintf(filename, sizeof(filename), "%s%s", root, "results/ads/adsorption.dat");
	if(restart)   adsorption = fopen(filename,"a");
	else          adsorption = fopen(filename,"w");
	
	if (adsorption == NULL) {
		printf("Error: file adsorption.dat could not be opened.\n");
		exit(1);
	}
	
	/** Create the file that contains the adsorption profiles in log-time */
	FILE * adsorptionLog;
	snprintf(filename, sizeof(filename), "%s%s", root, "results/ads/adsorptionLog.dat");
	if(restart)   adsorptionLog = fopen(filename,"a");
	else          adsorptionLog = fopen(filename,"w");
	
	if (adsorptionLog == NULL) {
		printf("Error: file adsorptionLog.dat could not be opened.\n");
		exit(1);
	}
	
	/** Create the file that contains the concentration profiles in linear time */
	FILE * concentration;
	snprintf(filename, sizeof(filename), "%s%s", root, "results/concentration/concentration.dat");
	if(restart)   concentration = fopen(filename,"a");
	else          concentration = fopen(filename,"w");
	
	if (concentration == NULL) {
		printf("Error: file concentration.dat could not be opened.\n");
		exit(1);
	}
	
	/** Create the file that contains the concentration profiles in log-time */
	FILE * concentrationLog;
	snprintf(filename, sizeof(filename), "%s%s", root, "results/concentration/concentrationLog.dat");
	if(restart)   concentrationLog = fopen(filename,"a");
	else          concentrationLog = fopen(filename,"w");
	
	if (concentrationLog == NULL) {
		printf("Error: file concentrationLog.dat could not be opened.\n");
		exit(1);
	}
	
	FILE * timeseries;
	snprintf(filename, sizeof(filename), "%s%s%d%s", root, "results/hist/tempHist_run", restartIndex, ".dat");
	timeseries = fopen(filename,"w");
	
	if (timeseries == NULL) {
		printf("Error: file tempHist_runX.dat could not be opened.\n");
		exit(1);
	}
	
	
	printf("Stop time: %ld\n", longestStopTime);
	
	for(int typei = 0; typei < ntypes; ++typei) 
		printf("Stop mol %d: %s\n", typei, (stopTimeMol[typei] > 0 ? "True" : "False"));
	
	for(int typei = 0; typei < ntypes; ++typei)
		printf("Stop time mol %d: %ld\n", typei, stopTimeMol[typei]);
	
	////////////////// SIMULATION STARTS HERE //////////////////////////
	
	if(constantConcentration) {
		util_applyBufferConditions();
		util_countInside();
		//util_setBuffer_Equil();
		util_setBuffer();
	}
	
	/// First check if we are restarting from a previous run or not
	if(!restart) {
		/// Add the first frame with the initial configuration to the movie file
		util_addXYZframe(movieLog);    
		util_addXYZframe(movie); 
		
		/////////////////////// SET VERLET LIST UPDATE PERIOD //////////////
		
		/// After equilibration, run for a number of cycles to compute the Verlet List update period
		verletlistcount = 0;
		
		////////////////// FAST EQUILIBRATE INIT CONFIG ////////////////////
		
		/** Run the simulation for a number of steps of equilibration, without adsorption */
		for (t = 0; t < equilTime; ++t) {
			/// OLD: Integrator with Berendsen thermostat to set T0 rapidly
			//verlet_integrateBerendsenNVT(); 
			verlet_integrateLangevinNVTequil();
			
			if(t % thermoSteps == 0) {
				
				/// Compute the thermodynamic quantities periodically
				/// Integrator + thermostat
				/// TODO: write new kernels that compute the virial and potential energy
				
				thermo_computeTemperature();
				
				Ekin = T * (kb * dim * N) * 0.5;
				
				if(verbose)
					printf("Step %lu Config \t Etot %1.4f Epot %1.4f Ekin %1.4f P %1.4f T %1.4f\n",abs(t),Etot/N,Epot/N,Ekin/N,P,T);
				
				fprintf(timeseries,"%lf %f\n",t*dt,T);
				
			}
			
			if(t % (keepRatioPeriod/10) == 0) {
				
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				
				
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				if(verbose)
					printf("Setting concentrations\n");
				
				util_countAdsorbed();
				util_countInside();
				if(constantConcentration)
					util_setBuffer();
			}
		}
		
		util_resetThermostat();
		
		fflush(timeseries);
		
		/// Period of Verlet List updates, no need to check maximum displacements anymore. Take a 66% for security
		updateVerletListPeriod = equilTime * .66 / verletlistcount;
		if(verbose)
			printf("listcouted %d times. Update period is: %d\n",verletlistcount,updateVerletListPeriod);
	}
	
	/////////////////////// MAIN SIMULATION LOOP ////////////////////////
	
	if (!restart && longestStopTime > 0) {
		
		logscale = 1.4;
		computeLogScale = (long int)(logscale + 0.5);
		
		
		for (t = 0; t < longestStopTime + 1; ++t) {
			
			verlet_integrateLangevinNVTrelax (t % updateVerletListPeriod == 0);
			
			if(t % thermoSteps == 0) {
				
				/// Compute the thermodynamic quantities periodically
				/// Integrator + thermostat
				/// TODO: write new kernels that compute the virial and potential energy
				
				thermo_computeTemperature();
				
				Ekin = T * (kb * dim * N) * 0.5;
				
				if(verbose)
					printf("Step %lu Config \t Etot %1.4f Epot %1.4f Ekin %1.4f P %1.4f T %1.4f\n",abs(t),Etot/N,Epot/N,Ekin/N,P,T);
				
				fprintf(timeseries,"%lf %f\n",t*dt,T);
				
			}
			
			
			if(t % keepRatioPeriod == 0) {
				
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				int adsorbed = util_countAdsorbed();
				
				if(verbose)
					printf("Adsorbed: %d\n",adsorbed);
				
				util_countInside();
				if(constantConcentration)
					util_setBuffer();
				util_fractionBound(t*dt);
				
				util_printConcentration(t*dt, concentration);
				
				/// Append to adsorption profile
				util_printAdsorbed(t*dt,adsorption);
			}
			
			/// Save movie frame. Warning! it is very slow and disk consuming
			if(t % moviePeriod == 0) {
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				util_addXYZframe(movie);
			}
			
			///save logscale data
			if(t == computeLogScale) {
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				int adsorbed = util_countAdsorbed();
				
				util_countInside();
				if(constantConcentration)
					util_setBuffer();
				util_fractionBound(t*dt);
				
				util_printConcentration(t*dt, concentrationLog);
				
				if(verbose)
					printf("Adsorbed: %d\n",adsorbed);
				
				/// Append to adsorption profile
				util_printAdsorbed(t*dt,adsorptionLog);
				
				/// Save movie frame. Warning! it is very slow
				util_addXYZframe(movieLog);
				
				logscale *= 1.4;
				computeLogScale = (long int)(offsetTime + logscale + 0.5);
			}
			
			
			/// Check if we reach the freezing time of any protein and need to de-freeze it
			for(int typei = 0; typei < ntypes; ++typei) {
				if(stopTimeMol[typei] > 0 && stopTimeMol[typei] == t) {
					
					//CudaSafeCall(cudaMemcpy( dev_stopMol,  stopMol,  N*sizeof(int),   cudaMemcpyHostToDevice));
					
					offsetTime = t;
					logscale = 1.4;
					computeLogScale = (long int)(offsetTime + logscale + 0.5);
					
					nTot[typei] = nRef[typei];
				}
			}
		}
		
		offsetTime = longestStopTime;
	}
	
	
	
	FILE * msdFile;
	bool computeMSD = FALSE;
	float4 * CoordIni = (float4*)malloc(N*sizeof(float4));
	
	if(!restart) {
		snprintf(filename, sizeof(filename), "%s%s", root, "results/msd/msd.dat");
		msdFile = fopen(filename,"w");
		computeMSD = TRUE;
		
		cudaMemcpy (CoordIni, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemset(dev_Image, 0, N*4*sizeof(int));
	}
	
	/// Logscale period of measures
	logscale = 1.4;
	computeLogScale = (long int)(offsetTime + logscale + 0.5);
	
	/// Flag that indicates if we need to fetch the coordinates from the GPU
	updateVerletList = offsetTime;
	
	printf("Starting main simulation\n");
	
	/** Run the simulation for a number of production cycles. starting from t = offsetTime */
	for (t = offsetTime; t < offsetTime + simulTime; ++t) 
	{
		verlet_integrateLangevinNVT (computeMSD, t == updateVerletList);
		
		if(t % thermoSteps == 0) {
			/// Integrator + thermostat
			/// TODO: write new kernels that compute the virial and potential energy!
			
			thermo_computeTemperature();
			Ekin = T * (kb*dim*N) * 0.5;
			
			if(verbose)
				printf("Step %lu Config \t Etot %1.4f Epot %1.4f Ekin %1.4f P %1.4f T %1.4f\n",t,Etot/N,Epot/N,Ekin/N,P,T);
			
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			
			/// Compute the number of adsorbed particles on-the-fly
			/// TDOD: THIS COULD ALSO BE DONE INSIDE THE GPU!
			int adsorbed = util_countAdsorbed();
			
			if(verbose)
				printf("Adsorbed: %d\n",adsorbed);
			
			/// Append to adsorption profile
			util_printAdsorbed(t*dt,adsorption);
			
			fprintf(timeseries,"%lf %f\n",t*dt,T);
		}
		
		///save logscale data
		if(t == computeLogScale) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			
			/// Compute the number of adsorbed particles on-the-fly
			/// TODO: THIS COULD ALSO BE DONE INSIDE THE GPU!
			int adsorbed = util_countAdsorbed();
			
			util_countInside();
			if(constantConcentration)
				util_setBuffer();
			util_fractionBound(t*dt);
			
			util_printConcentration(t, concentrationLog);
			
			if(verbose)
				printf("Adsorbed: %d\n",adsorbed);
			
			/// Append to adsorption profile
			util_printAdsorbed(t*dt,adsorptionLog);
			
			/// Save movie frame. Warning! it is very slow
			util_addXYZframe(movieLog);
			
			if(!restart) {
				cudaMemcpy (Image, dev_Image, N*sizeof(int4), cudaMemcpyDeviceToHost);
				
				float msd[ntypes];
				int num[ntypes];
				
				for(int typei = 0; typei < ntypes; ++typei) {
					msd[typei] = 0.;
					num[typei] = 0;
				}
				
				for(int i=0; i < N; ++i) {
					float dx = fabs(Coord[i].x + Image[i].x*Lx - CoordIni[i].x);
					float dy = fabs(Coord[i].y + Image[i].y*Ly - CoordIni[i].y);
					float dz = fabs(Coord[i].z + Image[i].z*Lz - CoordIni[i].z);
					
					msd[type[i]] += dx*dx + dy*dy + dz*dz;
					
					num[type[i]] ++;
				}
				
				for(int typei = 0; typei < ntypes; ++typei)
					msd[typei] /= num[typei];
				
				fprintf(msdFile,"%lf\t", (t-offsetTime+1)*dt);
				for(int typei = 0; typei < ntypes; ++typei)
					fprintf(msdFile,"%lf\t",msd[typei]);
				fprintf(msdFile,"\n");
				fflush(msdFile);
			}
			
			logscale *= 1.4;
			computeLogScale = (long int)(offsetTime + logscale + 0.5);
		}
		
		///save linear time data
		if(t % keepRatioPeriod == 0) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			
			/// Compute the number of adsorbed particles on-the-fly
			/// TODO: THIS COULD ALSO BE DONE INSIDE THE GPU!
			util_countAdsorbed();
			util_countInside();
			if(constantConcentration)
				util_setBuffer();
			
			util_printConcentration(t, concentration);
		}
		
		/// Save movie frame. Warning! it is very slow and disk consuming
		if(t % moviePeriod == 0) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			util_addXYZframe(movie);
		}
	}
	
	if(!restart)
		fclose(msdFile);
	
	fclose(movieLog);
	fflush(timeseries);
	
	/// Save the final configuration to file, in order to restart from it
	util_saveState(t);
	
	////////////// FINAL LOOP. COMPUTE RDF AND ENERGIES ////////////////
	
	printf("Finisihing simulation. Computing g(r)\n");
	for ( ; t < simulTime + offsetTime + rdfTime; ++t) 
	{
		/// Integrator + thermostat
		verlet_integrateLangevinNVT (0, t == updateVerletList);
		
		/// Compute the RDF at the end of the simulation, for a number of cycles
		if (t % rdfPeriod == 0) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			util_addToRdf();
		}
	}
	
	/// Compute the average and print the RDF to file
	util_calcPrintRdf(countRdf); 
	
	/// Free memory and close files
	fclose(movie);
	fclose(adsorption);
	fclose(timeseries);
	
	/// Reset the GPU to clear profiling settings or allocated memory for example
	CudaSafeCall(cudaDeviceReset());
	
	return 0;
}


__global__ void gpu_GenerateVerletListSync_kernel(
	char * type,
	float4 * dPos, 
	int * vlist, 
	unsigned char * nlist, 
	int N) 
{
	/// allocate shared memory
	extern __shared__ float4 sPos[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /** CONSTRAINT: threadIdMax must be a multiple of blockDim,
     * otherwise __synchthread() can not work! */
	/// save reference particle position in register
	float4 tPos = dPos[idx];               
	
	/// initialize the number of neighbors of reference particle
	unsigned char neig = 0;
	
	/// loop over all the blocks, that can synchronize
	for(int j=0; j<gridDim.x; ++j) 
	{
		/// offset particle id of current block
		int bOffset = j * blockDim.x;
		
		/// fetch block of particle positions and save in shared memory
		sPos[threadIdx.x] = dPos[bOffset + threadIdx.x];
		
		__syncthreads();
		
		/// loop over the particles in current block
		for (int i=0; i<blockDim.x; ++i) 
		{
			int id = bOffset + i;           /// current neighbor id, it is always id < N
			                                /// if id == idx, we count as neighbor, because pair_force(0) = 0 in the force table.
			
			float4 d = tPos - sPos[i];
			
			
			/// Image-corrected relative pair position 
			if(d.x > dev_hL)
				d.x -= dev_L;
			else if(d.x < - dev_hL)
				d.x += dev_L;
			
			if(d.y > dev_hL)
				d.y -= dev_L;
			else if(d.y < -dev_hL)
				d.y += dev_L;
			
			if(d.z > dev_hL)
				d.z -= dev_L;
			else if(d.z < -dev_hL)
				d.z += dev_L;
			
			
			/// neighbor list condition
			if (dev_vec_rv2[type[idx]] > d.x * d.x + d.y * d.y + d.z * d.z) {
                /// Save neighbor ID to the Verlet list and increase the number of neighbors
                vlist[neig * N + idx] = id;
                ++neig;
            }
		}
		
		/// Even if only one thread was succesful, all the other threads wait for it to finish
		__syncthreads();
	}
	
	/// check if we missed the last block of particles
	if(gridDim.x*blockDim.x < N) {
		
		int tOffset = gridDim.x*blockDim.x;
		
		if(tOffset+threadIdx.x < N)
			/// fetch block of particle positions and save in shared memory
			sPos[threadIdx.x] = dPos[tOffset+threadIdx.x]; 
		
		__syncthreads();
		
		/// loop over the particles in current block, until N
		for (int i = 0; i < N-tOffset; ++i) 
		{
			/// current neighbor id = [tOffset,N)
			int id = tOffset + i;
			
			float4 d = tPos - sPos[i];
			
			
			/// Image-corrected relative pair position 
			if(d.x > dev_hL)
				d.x -= dev_L;
			else if(d.x < - dev_hL)
				d.x += dev_L;
			
			if(d.y > dev_hL)
				d.y -= dev_L;
			else if(d.y < -dev_hL)
				d.y += dev_L;
			
			if(d.z > dev_hL)
				d.z -= dev_L;
			else if(d.z < -dev_hL)
				d.z += dev_L;
			
			
			/// neighbor list condition
			if ( dev_vec_rv2[type[idx]] > d.x*d.x + d.y*d.y + d.z*d.z ) 
			{
				/// save neighbor id to verlet list and increase the number of neighbors
				vlist[neig*N + idx] = id;
				++ neig;
			}
		}
		
		/// Even if only one thread was succesful, all the other threads wait for it to finish
		__syncthreads();
	}
	
	/// save the number of neighbors to global memory, colaesced
	nlist[idx] = neig;
}

/// UNDER CONSTRUCTION ///
__global__ void gpu_GenerateVerletListAsync_kernel(
	float4 * Coord, 
	int * vlist, 
	unsigned char * nlist, 
	int N,
	int offset) 
{
	int idx = offset + threadIdx.x;
       
	/// save reference particle position in register
	float4 tPos = Coord[idx];
	unsigned char neig = 0;
	
	for(int i = 0; i < N; ++ i)
	{
		float4 d = tPos - Coord[i];
		
		/// Image-corrected relative pair position 
		if(d.x > dev_hL)
			d.x -= dev_L;
		else if(d.x < - dev_hL)
			d.x += dev_L;
		
		if(d.y > dev_hL)
			d.y -= dev_L;
		else if(d.y < -dev_hL)
			d.y += dev_L;
		
		if(d.z > dev_hL)
			d.z -= dev_L;
		else if(d.z < -dev_hL)
			d.z += dev_L;
		 
		if (dev_rv2 > d.x*d.x + d.y*d.y + d.z*d.z) {
			vlist[neig*N + idx] = i;
			++ neig;
		}
	 }
	
	__syncthreads();
	
	/// save the number of neighbors to global memory, colaesced
	nlist[idx] = neig;
}

void util_calcVerletList (void) 
{
	int nthreads = NTHREADS;                   /// Number of launch threads
	int nblocks = N / nthreads;                /// Only Synchronous blocks
	int cacheMem = nthreads * sizeof(float4);  /// Amount of shared memory
	
	++verletlistcount;
	updateVerletList += updateVerletListPeriod;
	
	/** cal indiciar el nombre de: 
	 * <<< threads, blocks, memoria cache = numThreadsPerBlock*sizeof(float4) >>> */
	gpu_GenerateVerletListSync_kernel <<<nblocks, nthreads, cacheMem>>> (dev_type,dev_Coord,dev_vlist,dev_nlist,N); 
	if(debug)
		CudaCheckError();
	
	/// Check for threads in Asynchronous blocks
	if(nblocks*nthreads < N)
	{
		gpu_GenerateVerletListAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Coord, dev_vlist, dev_nlist, N, nblocks*nthreads);
		if(debug)
		CudaCheckError();
	}
	
	/// Update verlet lists in GPU mem
	CudaSafeCall(cudaMemcpy( dev_CoordVerlet, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToDevice ));
	

	/// Security check, optional but highly recomendable!
	CudaSafeCall(cudaMemcpy( nlist, dev_nlist, N*sizeof(unsigned char), cudaMemcpyDeviceToHost ));
	
	int i,max = 0;
	
	for(i = 0; i < N; ++i) 
		if(nlist[i] > max)
			max = nlist[i];
	
	if(max > MAXNEIG) 
	{
		printf("////////////// Fail!!! ///////////\n");
		printf("Too many neighbors in Verlet List!\n");
		printf("////////////// Fail!!! ///////////\n\n");
		
		exit(1);
	}
	
	neigmax = max;
}


/// This function calculates the temperature of the system 
double thermo_temperature (void) 
{
	int i;
	double temp, v2;
	
	temp = 0;
	
	for (i = 0; i < N; ++i) 
	{
		v2 = Vel[i].x*Vel[i].x + Vel[i].y*Vel[i].y + Vel[i].z*Vel[i].z;
		
		temp += M[type[i]] * v2;
	}
	
	/// Compute the kinetic energy
	Ekin = 0.5 * temp;
	
	/// And the temperature
	temp /= kb * dim * (N - 1);
	
	return temp;
}

/// Box-Muller algorithm to generate gaussian randoms
double util_gaussianRandom (double mu, double sig)
{
	double u1 = rand()/(RAND_MAX + 1.);
	double u2 = rand()/(RAND_MAX + 1.);
	
	double z1 = sqrt(-2 * log(u1)) * sin(2 * M_PI * u2);
	
	return mu + z1 * sig;
}

double util_uniformRandom (double mu, double sig)
{
	/// sqrt(12) = 3.4641
	double u = 3.4641 * (rand()/(RAND_MAX + 1.) - 0.5);
	
	return mu + u * sig;
}

void util_generateRandomForces (void) 
{
	
	double coeff = sqrt(2*T0/dt);
	int i;
	
	for(i = 0; i < N; ++i) {
		RandomForce[i] = coeff*sqrt(Gamma[i]) / sqrM[type[i]] * make_float4(util_uniformRandom(0,1),util_uniformRandom(0,1),util_uniformRandom(0,1),0);
	}
	
	CudaSafeCall(cudaMemcpy( dev_Rnd, RandomForce, N*sizeof(float4), cudaMemcpyHostToDevice ));
}




__global__ void gpu_RNG_setup ( 
	curandState * state, 
	unsigned long seed, 
	int N )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(id < N) {
	
	curand_init( (seed << 20) + id, 0, 0, &state[id]);
	
	id += blockDim.x*gridDim.x;
    }
} 

__global__ void gpu_RNG_generate ( 
	curandState* globalState, 
	float4 * Rnd, 
	char * type,
	int N ) 
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(ind < N) {
	    
	    int typei = type[ind];
	    
	    __syncthreads();
	    
	    curandState localState = globalState[ind];
	    
	    /// uniform random in interval [-0.5:0.5] with zero mean and variance 1
	    
	    float4 randNum;
	    
	    randNum.x = curand_uniform( &localState ) - 0.5f;
	    randNum.y = curand_uniform( &localState ) - 0.5f;
	    randNum.z = curand_uniform( &localState ) - 0.5f;
	    randNum *= 3.4641f;
	    
	    randNum *= dev_langPrefactor[typei] / dev_sqrM[typei];
	    
	    __syncthreads();
	    
	    Rnd[ind] = randNum;
	    
	    globalState[ind] = localState; 
	    
	    ind += blockDim.x*gridDim.x;
	}
}


void init_config(char * filename, int filenameln) 
{
	double xi,yi,zi,Mi,Ri,Rsi,epsi,epsi2,epsi3;
	int i;
	char typei;
	
	FILE *file;
	
	struct  {
		double x;
		double y;
		double z;
	} momentum;
	
	momentum.x = momentum.y = momentum.z = 0;
	
	printf("setup.dat: %s\n", filename);

	file = fopen(filename,"r");
	
	if (file == NULL) {
		printf("Error: file setup.dat not found.\n");
		exit(1);
	}
	
	char * line = NULL;
	char str[100];
	size_t len = 0;
	
	stopTimeMol = (long int *)calloc(sizeof(long int), ntypes);
	
	/// Option parser
	while (getline(&line, &len, file) != -1) {
		
		sprintf(str,"");
		sscanf (line,"%s %*s", str);
		
		/// skip commented lines
		if(str[0] == '#')
			continue;
		
		if (!strcmp(str, "moviePeriod")) {
			sscanf (line,"%s %ld", str, &moviePeriod);
			printf("number of movie period steps set to: %ld\n", moviePeriod);
		}
		else if (!strcmp(str, "thermoSteps")){
			sscanf (line,"%s %ld", str, &thermoSteps);
			printf("number of thermo steps set to: %ld\n", thermoSteps);
		}
		else if (!strcmp(str, "keepRatioPeriod")){
			sscanf (line,"%s %ld", str, &keepRatioPeriod);
			printf("number of keep ratio steps set to: %ld\n", keepRatioPeriod);
		}
		else if (!strcmp(str, "rdfPeriod")){
			sscanf (line,"%s %ld", str, &rdfPeriod);
			printf("number of rdf period steps set to: %ld\n", rdfPeriod);
		}
		else if (!strcmp(str, "equilTime")){
			sscanf (line,"%s %ld", str, &equilTime);
			printf("number of equilibration timesteps set to: %ld\n", equilTime);
		}
		else if (!strcmp(str, "nosehooverTime")){
			sscanf (line,"%s %ld", str, &nosehooverTime);
			printf("number of Nose-Hoover thermostat timesteps set to: %ld\n", nosehooverTime);
		}
		else if (!strcmp(str, "relaxTime1")){
			sscanf (line,"%s %ld", str, &relaxTime1);
			printf("number of 1st relaxation time set to: %ld\n", relaxTime1);
		}
		else if (!strcmp(str, "relaxTime2")){
			sscanf (line,"%s %ld", str, &relaxTime2);
			printf("number of 2nd relaxation time set to: %ld\n", relaxTime2);
		}
		else if (!strcmp(str, "stopTimeMol1")){
			sscanf (line,"%s %ld", str, &stopTimeMol[0]);
			printf("Number mol 1 freezing timesteps: %ld\n", stopTimeMol[0]);
		}
		else if (!strcmp(str, "stopTimeMol2")){
			sscanf (line,"%s %ld", str, &stopTimeMol[1]);
			printf("Number mol 2 freezing timesteps: %ld\n", stopTimeMol[1]);
		}
		else if (!strcmp(str, "stopTimeMol3")){
			sscanf (line,"%s %ld", str, &stopTimeMol[2]);
			printf("Number mol 3 freezing timesteps: %ld\n", stopTimeMol[2]);
		}
		else if (!strcmp(str, "simulTime")){
			sscanf (line,"%s %ld", str, &simulTime);
			printf("number of simulation timesteps set to: %ld\n", simulTime);
		}
		else if (!strcmp(str, "rdfTime")){
			sscanf (line,"%s %ld", str, &rdfTime);
			printf("number of rdf measure timesteps set to: %ld\n", rdfTime);
		}
		
		else if (!strcmp(str, "dt")) {
			sscanf (line,"%s %f", str, &dt);
			hdt = 0.5 * dt;
			printf("Timestep set to dt=%f\n", dt);
		}
		else if (!strcmp(str, "Tinit")) {
			sscanf (line,"%s %f", str, &Tinit);
			printf("Starting temperature set to Tinit=%f\n", Tinit);
		}
		else if (!strcmp(str, "T0")) {
			sscanf (line,"%s %f", str, &T0);
			printf("Thermostat objective temperature set to T0=%f\n", T0);
		}
		else if (!strcmp(str, "Gamma1")) {
			sscanf (line,"%s %f", str, &Gamma1);
			printf("Langevin Heat-bath coupling Gamma1=%f\n", Gamma1);
		}
		else if (!strcmp(str, "Gamma2")) {
			sscanf (line,"%s %f", str, &Gamma2);
			printf("Langevin Heat-bath coupling Gamma2=%f\n", Gamma2);
		}
		else if (!strcmp(str, "Gamma3")) {
			sscanf (line,"%s %f", str, &Gamma3);
			printf("Langevin Heat-bath coupling Gamma3=%f\n", Gamma3);
		}
		
		else if (!strcmp(str, "restart")) {
			restart = 1;
			printf("Restarting from previous run\n");
		}
		
		else if (!strcmp(str, "constantConcentration")) {
			constantConcentration = TRUE;
			printf("Keeping constant concentration of proteins\n");
		}
		
		else if (!strcmp(str, "gpuId")) {
			sscanf (line,"%s %d", str, &gpuId);
			/// Set the GPU we want to use, default 0 or set by configuration file
			printf("Using GPU number %d\n", gpuId);
		}
		
		else if (!strcmp(str, "nThreads")) {
			sscanf (line,"%s %d", str, &NTHREADS);
			/// Set the number on threads to use in each GPU kernel
			printf("Number of GPU threads set to %d\n", NTHREADS);
		}
		
		else if (!strcmp(str, "verbose")) {
			verbose = 1;
			printf("Verbose mode activated\n");
		}
		else if (!strcmp(str, "root")) {
			sscanf (line,"%s %s", str, root);
			printf("root:%s\n", root);
		}
		else
			if(strlen(str) > 1) {
				printf("**********************************************\n");
				printf("!!! setup.dat Read Error, unkown option: %s \n", str);
				printf("**********************************************\n");
				exit(1);
			}
	}
	
	fclose(file);
	
	CudaSafeCall(cudaSetDevice(gpuId));
	
	
	/// Here we open the file generated by the script
	snprintf(filename, filenameln, "%s%s", root, "config/init.dat");
	printf("init.dat: %s\n", filename);
	file = fopen(filename,"r");
	
	if (file == NULL) {
		printf("Error: file config/init.dat not found.\n");
		exit(1);
	}
	
	float boxExp;
	
	/// Read the first line of the file input.dat
	fscanf(file,"%d\t%lf\t%lf\t%d\t%lf\t%f\t%f\t",&N,&C_PBS,&K_pp,&epsDiag,&box,&V,&boxExp);
	for(typei = 0; typei<ntypes; ++typei) {
		fscanf(file,"%d\t",&(nRef[typei]));
		printf("nRef %d: %d\n", typei, nRef[typei]);
	}
	fscanf(file,"\n");
	
	printf("K_pp: %lf\n", K_pp);
	
	printf("EpsDiag: %s\n", epsDiag == 1 ? "ON" : "OFF");
	
	//K2 = K_pp*K_pp;
	K = K_pp;
	
	/// Debye-H"ukel screening length ([C_PBS] = M = 1 mol/l)
	kappa = 5.08 * sqrt(C_PBS);
	
	printf("Debye length: %lf\n", kappa);
	
	/// Reference Experimental values of the concentrations
	/// Volume of a box containing exactly 1 NP
	VTot = pow(boxExp, 3); 
	
	
	/// Remove the NP from the number of particles
	N -= 1;
	
	int npart = N;
	
	/// Reconfigure the thermostat parameters
	Q0 = (20*dt*20*dt*npart*T0);
	Q1 = (Q0/npart);
	Q2 = (Q1/4);
	tau = 1./(20*dt);
	
	/// Set the box dimensions
	Lx = box;
	Ly = box;
	Lz = box;
	
	L = box;
	
	hLx = Lx*0.5;
	hLy = Ly*0.5;
	hLz = Lz*0.5;
	
	hL = L*0.5;
	
	hbox = box*0.5;
	
	/// Array with the type of each particle
	type = (char *)malloc(npart*sizeof(char));
	CudaSafeCall(cudaMalloc( (void**)&dev_type, npart*sizeof(char) ));
	
	isOutsideClosed = (int *)calloc(N, sizeof(int));
	CudaSafeCall(cudaMalloc( (void**)&dev_isOutsideClosed, N*sizeof(int) ));
	
	//stopMol = (int *)calloc(N, sizeof(int));
	//CudaSafeCall(cudaMalloc( (void**)&dev_stopMol, N*sizeof(int) ));
	
	/// Array with the mass of each type
	M = (float *)malloc(ntypes*sizeof(float));
	sqrM = (float *)malloc(ntypes*sizeof(float));
	
	/// Arrays with the radiuses of each type
	R = (float *)malloc(ntypes*sizeof(float));
	Rs = (float *)malloc(ntypes*sizeof(float));
	
	Gamma = (float *)malloc(npart*sizeof(float));
	CudaSafeCall(cudaMalloc( (void**)&dev_gamma, npart*sizeof(float) ));
	
	/// Arrays with the affinity of each type with the NP
	EpsNP = (float *)malloc(ntypes*sizeof(float));
	
	EpsEDL = (float *)malloc(ntypes*sizeof(float));
	
	EpsProt = (float *)malloc(ntypes*sizeof(float));
	
	/// Arrays with the coordinates of the particles
	Coord = (float4 *)malloc(npart*sizeof(float4));
	CudaSafeCall(cudaMalloc( (void**)&dev_Coord, npart*sizeof(float4) ));
	
	/// Arrays with the periodic image of the particles
	Image = (int4 *)malloc(npart*sizeof(int4));
	cudaMalloc( (void**)&dev_Image, npart*sizeof(int4) );
	
	/// Arrays with the velocities of the particles
	Vel = (float4 *)malloc(npart*sizeof(float4));
	CudaSafeCall(cudaMalloc( (void**)&dev_Vel, npart*sizeof(float4) ));
	
	/// Arrays with the accelerations of the particles
	Acc = (float4 *)malloc(npart*sizeof(float4));
	CudaSafeCall(cudaMalloc( (void**)&dev_Acc, npart*sizeof(float4) ));
	
	/// Arrays with the coordinates of the verlet coordinates
	CoordVerlet = (float4 *) malloc(npart*sizeof(float4));
	CudaSafeCall(cudaMalloc( (void**)&dev_CoordVerlet, npart*sizeof(float4) ));
	
	RandomForce = (float4 *)malloc(npart*sizeof(float4));
	CudaSafeCall(cudaMalloc( (void**)&dev_Rnd, npart*sizeof(float4) ));
	
	/// Some thermodynamic quantities
//	partial_Epot = (float *)malloc(npart*sizeof(float));
	CudaSafeCall(cudaMalloc( (void**)&dev_partial_Epot, npart*sizeof(float) ));
	
//	partial_P = (float *)malloc(npart*sizeof(float));
	CudaSafeCall(cudaMalloc( (void**)&dev_partial_P, npart*sizeof(float) ));
	
	CudaSafeCall(cudaMalloc( (void**)&dev_Epot, sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_P, sizeof(float) ));
	
	CudaSafeCall(cudaMalloc( (void**)&dev_Epot_NP, sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_P_NP, sizeof(float) ));
	
	CudaSafeCall(cudaMalloc( (void**)&dev_Epot_walls, sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_P_walls, sizeof(float) ));
	
	CudaSafeCall(cudaMalloc( (void**)&dev_partialT, 64*sizeof(double) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_T, sizeof(double) ));
	
	CudaSafeCall(cudaMalloc ( &devStates, npart*sizeof( curandState ) ));
	
	/// seed = time(NULL) or seed = 1234 for debugging
	gpu_RNG_setup <<< (npart + NTHREADS - 1) / NTHREADS, NTHREADS >>> ( devStates, 1234 , npart);
	
	chi0 = chi1 = chi2 = 0.;
	
	for(typei = 0; typei < ntypes; ++typei)
		R[typei] = Rs[typei] = 0.;
	
	int * num = (int*)calloc(sizeof(int), ntypes);
	
	/// Read the init file
	for(i = 0; i < npart; ++i) 
	{
		fscanf(file,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&typei,&xi,&yi,&zi,&Ri,&Rsi,&Mi,&epsi,&epsi2,&epsi3);
		
		type[i] = typei;
		//printf("Type[i] %d;",type[i]);
		num[typei] ++;
		
		/// Count how many particles of each type
		float r = sqrt((xi - hbox)*(xi - hbox) + (yi - hbox)*(yi - hbox) + (zi - hbox)*(zi - hbox));
		
		Coord[i] = make_float4(xi,yi,zi,0.f);
		
		R[typei] = Ri;
		Rs[typei] = Rsi;
		
		M[typei] = Mi;
		sqrM[typei] = sqrt(Mi);
		EpsNP[typei] = epsi;
		EpsEDL[typei] = epsi2;
		EpsProt[typei] = epsi3;
		if(typei==0) Gamma[i] = Gamma1;
		else if(typei==1) Gamma[i] = Gamma2;
		else if(typei==2) Gamma[i] = Gamma3;

		
		Acc[i] = make_float4(0.,0.,0.,0.);
		
		Vel[i].x = util_gaussianRandom(0,1) * sqrt(kb * Tinit / Mi);
		Vel[i].y = util_gaussianRandom(0,1) * sqrt(kb * Tinit / Mi);
		Vel[i].z = util_gaussianRandom(0,1) * sqrt(kb * Tinit / Mi);
		
		momentum.x += Mi * Vel[i].x;
		momentum.y += Mi * Vel[i].y;
		momentum.z += Mi * Vel[i].z;
	}
	
	momentum.x /= npart;
	momentum.y /= npart;
	momentum.z /= npart;
	
	/// Transform all the velocities (except the NP) to zero total momentum 
	for (i = 0; i < npart; ++i) 
	{
		Vel[i].x -= momentum.x/M[type[i]];
		Vel[i].y -= momentum.y/M[type[i]];
		Vel[i].z -= momentum.z/M[type[i]];
	}
	
	/// Read NP radius
	fscanf(file,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&typei,&xi,&yi,&zi,&Ri,&Rsi,&Mi,&epsi,&epsi2);
	R_NP = Ri;
	
	if(verbose)
		printf("Parameter R_NP = %f\n",R_NP);
	
	x_NP = y_NP = z_NP = hbox;
	
	/// Number of proteins inside a box of volume Vtot
	for(typei = 0; typei < ntypes; ++typei)
		nTot[typei] = nRef[typei];
	
	for (i = 0; i < N; ++i) {
		typei = type[i];
		
		if (stopTimeMol[typei] > 0) {
			nTot[typei] = 0;
			//stopMol[i] = 1;
			if (stopTimeMol[typei] > longestStopTime)
				longestStopTime = stopTimeMol[typei];
		}
	}
	
	/// Look for the maximum radius of the proteins
	float maxR = 0.;
	
	for(i = 0; i < ntypes; ++i)
		if(num[i] > 0 && maxR < Rs[i])
			maxR = Rs[i];
	printf("maxR %f Rs:%f,%f,%f\n",maxR,Rs[0],Rs[1],Rs[2]);
	/// Cutoff d'interaccio proteina-proteina
	for(typei = 0; typei < ntypes; ++typei){
		vec_rc2[typei] = (Rs[typei] + maxR) + Rs[typei];
		vec_rc2[typei] *= vec_rc2[typei];
	}
	
	maxR = 0.;
	
	for(i = 0; i < ntypes; ++i)
		if(num[i] > 0 && maxR < vec_rc2[i])
			maxR = vec_rc2[i];
	
	rc = sqrt(maxR);
	
	if(verbose) {
		printf("Parameter cutoff radius ");
		for(typei = 0; typei < ntypes; ++typei)
			printf("rc%d: %1.3f, ", typei, sqrt(vec_rc2[typei]));
		printf("\n");
	}
	
	rc2_NP = 5 * (2*rc);
	rc2_NP *= rc2_NP;
	
	/// Radi de l'esfera de verlet que conte els veins
	for(typei = 0; typei < ntypes; ++typei)
		vec_rv2[typei] = vec_rc2[typei] * 1.5*1.5;
	
	if(verbose) {
		printf("Parameter verlet radius ");
		for(typei = 0; typei < ntypes; ++typei)
			printf("rc%d: %1.3f, ", typei, sqrt(vec_rv2[typei]));
		printf("\n");
	}
	
	double minR = 1000.;
	int minRt;
	
	for(i = 0; i < ntypes; ++i)
		if(num[i] > 0 && minR > sqrt(vec_rv2[i])) {
			minR = sqrt(vec_rv2[i]);
			minRt = i;
		}
	
	/// Amplada de la 'pell' de l'esfera de verlet, que determina la frequencia amb la qual es recalcula la llista. Prenem el tipus de proteina mes petita, i mes rapida
	skin2 = (sqrt(vec_rv2[minRt]) - sqrt(vec_rc2[minRt]))*0.5;
	skin2 *= skin2;
	
	if(verbose)
		printf("Parameter verlet skin: %1.3f\n", skin2);
	
	
	/// Allotjament de les llistes de verlet i altres quantitats relacionades
	nlist  = (unsigned char *)malloc(npart*sizeof(unsigned char));
	CudaSafeCall(cudaMalloc( (void **)&dev_nlist, npart*sizeof(unsigned char)));
	
	if(verbose) {
		printf("Neighbor list allocated\n");
		fflush(stdout);
	}
	
	vlist = (int *)malloc(npart*MAXNEIG*sizeof(int));
	CudaSafeCall(cudaMalloc( (void **)&dev_vlist, npart*MAXNEIG*sizeof(int)));
	
	if(verbose) {
		printf("Verlet list allocated\n");
		fflush(stdout);
	}
	
	CudaSafeCall(cudaMalloc( (void **)&dev_neigmax, sizeof(unsigned char)));
	CudaSafeCall(cudaMalloc((void **)&dev_newlist,sizeof(unsigned int)));
	
	if(verbose) {
		printf("GPU Neighbor parameters allocated\n");
		fflush(stdout);
	}
	
	fclose(file);
}

/// *** STILL UNDER CONSTRUCTION *** ///
void util_loadConfig (void) 
{
	double xi,yi,zi,vx,vy,vz,ax,ay,az,Mi,Ri,Rsi,epsi,epsi2,epsi3;
	int i;
	char typei;
	
	FILE *file;
	char filename[100];
	
	/// Here we open the file generated by this program in a previous run
	snprintf(filename, sizeof(filename), "%s%s", root, "config/lastconfig.dat");
	file = fopen(filename,"r");
	
	if (file == NULL) {
		printf("Error: file config/lastconfig.dat could not be opened. You must first run this program without the 'restart' option.\n");
		exit(1);
	}
	fscanf(file, "%lu\t%d\t%lf\t%d\n", &offsetTime, &updateVerletListPeriod, &logscale, &restartIndex);
	
	++restartIndex;
	
	/// Read the init file
	for(i = 0; i < N; ++i) 
	{
		fscanf(file,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
		&typei,
		&xi, &yi, &zi,
		&vx, &vy, &vz, 
		&ax, &ay, &az, 
		&Ri, &Rsi, &Mi, 
		&epsi, &epsi2, &epsi3);
		
		type[i] = typei;
		
		R[typei]       = Ri;
		Rs[typei]      = Rsi;
		M[typei]       = Mi;
		sqrM[typei]    = sqrt(Mi);
		EpsNP[typei]   = epsi;
		EpsEDL[typei]   = epsi2;
		EpsProt[typei] = epsi3;
		
		Coord[i] = make_float4(xi, yi, zi, 0.f);
		Vel[i]   = make_float4(vx, vy, vz, 0.f);
		Acc[i]   = make_float4(ax, ay, az, 0.f);
	}
	
	x_NP = y_NP = z_NP = hbox;
	
	
	fclose(file);
}


__global__ void gpu_divideAccMass_kernel (char * type, float4 * Acc, int N) 
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	
	while(i < N)
	{
		Acc[i] /= dev_M[type[i]];
		
		i += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_addRandomForces_kernel (
	float4 * Acc, 
	float4 * Rnd, 
	int N ) 
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	
	while(i < N)
	{
		Acc[i] += Rnd[i];
		
		i += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_forceThreadPerAtomTabulatedEquilSync_kernel (
	char * type,
	float4 * Coord,
	float4 * Acc,
	float * tableF_rep,
	unsigned char * nlist, 
	int * vlist,
	int N) 
{
	/// Cada thread s'encarrega del calcul de una particula
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	/// number of neighbors of current particle, coalesced read
	unsigned char neighbors = nlist[i];
	
	/// eficient read, stored in texture cache for later scattered reads
	float4 ri = Coord[i];
	
	int typei = type[i];
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		int j = vlist[neig*N + i];
		
		/// Distance between 'i' and 'j' particles. RANDOM read
		float4 rj = Coord[j];
		float4 rij = rj - ri;
		
		rij.x -= dev_L * rintf(rij.x / dev_L);
		rij.y -= dev_L * rintf(rij.y / dev_L);
		rij.z -= dev_L * rintf(rij.z / dev_L);
		
		/// Save r2 = r*r in rij.w
		rij.w = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
		
		/// Skip current particle if not witihin cutoff
		if ( rij.w < dev_vec_rc2[typei])
		{
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			/// Compute the acceleration in each direction
			/// Nomes per a les acceleracions de la particula del block actual
			/// Falta dividir per la massa
			acc += - tableF_rep[(int)(rij.w*100 + 0.5f) + dev_ntable*(type[j] + ntypes*typei)] * rij;
		}
	}
	
	/// coalesced write to global memory
	Acc[i] = acc;
}




__global__ void gpu_forceThreadPerAtomTabulatedSync_kernel (
	char * type, 
	float4 * Coord,
	float4 * Acc,
	float * tableF_rep,
	float * tableF_att_symm,
	float * tableF_att_noSymm,
	unsigned char * nlist, 
	int * vlist,
	int N) 
{
	/// Cada thread s'encarrega del calcul de una particula
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	/// number of neighbors of current particle, coalesced read
	unsigned char neighbors = nlist[i];
	
	/// eficient read, stored in texture cache for later scattered reads
	float4 ri = Coord[i];
	
	int typei = type[i];
	
/// Activate with soft-corona interactions
#ifdef SC
	float4 di = ri - dev_xyzNP;
	/// Distance to the surface of the NP
	di.w = sqrtf(di.x*di.x + di.y*di.y + di.z*di.z) - dev_R_NP;
#endif
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		int j = vlist[neig*N + i];
		
		 /// Distance between 'i' and 'j' particles. RANDOM read
		float4 rj = Coord[j];
		float4 rij = rj - ri;
		
		float pair_force_symm = 0.f;
#ifdef SC
		float pair_force_asymm = 0.f;
#endif
		
		
		rij.x -= dev_L * rintf(rij.x / dev_L);
		rij.y -= dev_L * rintf(rij.y / dev_L);
		rij.z -= dev_L * rintf(rij.z / dev_L);
		
		/// Save r2 = r*r in rij.w
		rij.w = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
		
		/// Skip current particle if not witihin cutoff
		if ( rij.w < dev_vec_rc2[typei])
		{
			int typej = type[j];
			
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			pair_force_symm = - tableF_rep[(int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)];
			
#ifdef SC
			float4 dj = rj - dev_xyzNP;
			dj.w = sqrtf(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z) - dev_R_NP;
			
			/// Attractive interaction force, depends on the distances of i and j to te NP center
			/// Symmetric part
			float expo = exp(-di.w*dj.w/(dev_K*dev_K));
			
			pair_force_symm += expo * tableF_att_symm[(int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)];
			
			/// Non-symmetric part
			pair_force_asymm = - expo*di.w/(dev_K*dev_K) * tableF_att_noSymm[(int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)];
#endif
		
		/// Compute the acceleration in each direction
		/// Nomes per a les acceleracions de la particula del block actual
		/// Falta dividir per la massa
		acc += pair_force_symm * rij;
#ifdef SC
		/// d/(d + Rnp) is a unitary vector in the direction of ri-Rnp
		acc += pair_force_asymm * di/(di.w+dev_R_NP);
#endif
		}
	}
	
	/// coalesced write to global memory
	Acc[i] = acc;
}


/// *** STILL UNDER CONSTRUCTION *** ///
__global__ void gpu_forceThreadPerAtomTabulatedAsync_kernel (
	char * type,
	float4 * Coord,
	float4 * Acc,
	float * tableF_rep,
	unsigned char * nlist, 
	int * vlist,
	int N,
	int offset) 
{
	/// Cada thread s'encarrega del calcul de una particula
	int i = offset + threadIdx.x;
	
	float pair_force;
	
	/// number of neighbors of current particle, coalesced read
	unsigned char neighbors = nlist[i];
	
	int typei = type[i];
	
	/// eficient read, stored in texture cache for later scattered reads
	float4 ri = Coord[i];
	
	float4 di = ri - dev_xyzNP;
	/// Distance to the center of the NP
	di.w = sqrt(di.x*di.x + di.y*di.y + di.z*di.z) - dev_R_NP;
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		//int j = vlist[neig*N + i];
		
		/// Distance between 'i' and 'j' particles. WARNING: Scattered (random) read
		float4 rj = Coord[vlist[neig*N + i]]; 
		float4 rij = ri - rj;
		
		/// Image-corrected relative pair position 
		if(rij.x > dev_hL)
			rij.x -= dev_L;
		else if(rij.x < - dev_hL)
			rij.x += dev_L;
		
		if(rij.y > dev_hL)
			rij.y -= dev_L;
		else if(rij.y < -dev_hL)
			rij.y += dev_L;
		
		if(rij.z > dev_hL)
			rij.z -= dev_L;
		else if(rij.z < -dev_hL)
			rij.z += dev_L;
		
		 /// Save r2 = r*r in rij.w
		rij.w = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
		
		/// Skip current particle if not witihin cutoff
		if(rij.w < dev_vec_rc2[typei])
		{
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			pair_force = tableF_rep[(int)(rij.w*100 + 0.5f)];
#ifdef SC
			float4 dj = rj - dev_xyzNP;
			dj.w = sqrtf(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z) - dev_R_NP;
			
			/// Attractive interaction force, depends on the distances of i and j to te NP center
			//pair_force += (sqrtf(1.125f*1.125f/(di.w*dj.w))/* + 1.125*1.125/(dj.w*dj.w)*/) * tex1Dfetch(tableF_att_tex,(int)(rij.w/dev_dr2));
//			pair_force += (sqrtf(1.125f*1.125f/(di.w*dj.w))/* + 1.125*1.125/(dj.w*dj.w)*/) * tex1Dfetch(tableF_att_tex,(int)(rij.w*100000 + 0.5f));
#endif
		}
		else
			pair_force = 0.f;
		
		/// Compute the acceleration in each direction
		/// Nomes per a les acceleracions de la particula del block actual
		/// Falta dividir per la massa
		acc += pair_force * rij;
	}
	
	__syncthreads();
	
	/// coalesced write to global memory
	Acc[i] = acc;
	
}

/// NOT IN USE
__device__ float gpu_calcPNPForce (float eps, float sigma2, float r2) 
{
	/// (sigma/r)**2
	r2 = sigma2 / r2;
	/// (sigma/r)**4
	float r12 = r2*r2;
	/// (sigma/r)**12
	r12 = r12*r12*r12;
	
	return 4.f*24.f * eps * r2 * r12 * (r12 - 0.5) / sigma2;
}

/// NOT IN USE
__device__ float gpu_calcPNPForceEquil (float eps, float sigma2, float r2) 
{
	/// (sigma/r)**2
	r2 = sigma2 / r2;
	/// (sigma/r)**4
	float r12 = r2*r2;
	/// (sigma/r)**12
	r12 = r12*r12*r12;
	
	/// Only repulsive, no adsorption possible
	return 4.f*24.f * eps * r2 * r12 * r12 / sigma2;
}

/// NOT IN USE
__global__ void gpu_forceNP_kernel (char * type, float4 * Coord, float4 * Acc, int N)
{
	float r2, r;
	float4 acc, Coords;
	float pair_force;
	float Ri, EpsNPi;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int threadIdMax = N/blockDim.x * blockDim.x;
	
	/// Calculem la interaccio de cada particula amb la NP
	while (i < N) 
	{
		if(i < threadIdMax)
			__syncthreads();
		
		acc = make_float4(0.f,0.f,0.f,0.f);
		
		/// Type of the current particle
		int typei = type[i];
		
		/// Properties of the given type
		EpsNPi = dev_EpsNP[typei];
		Ri = dev_R[typei];
		
		/// Read partice positions. Coalesced read
		Coords = Coord[i];
		
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		/// Distance to te NP
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;
		 
		/// Distance between the center of a protein and the NP
		r = sqrt(r2);
		r -= dev_R_NP;
		
		/// Rescale positions to shift the interaction potential
		Coords *= r/sqrt(r2);
		
		/// Squared distance of a protein to the NP's surface
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;
		
		if(r2 < dev_rc2_NP)
		{
			pair_force = gpu_calcPNPForce(EpsNPi, Ri*Ri, r2);
			
			/// Update particle accelerations
			acc = pair_force * Coords;
		}
		
		if(i < threadIdMax)
			__syncthreads();
		
		/// Coalesced write
		Acc[i] += acc;
		/// Compute the resulting acceleration: a = F / m
		Acc[i] /= dev_M[typei];
		
		/// Next particle
		i += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_forceNP_DLVO_kernel (char * type, float4 * Coord, float4 * Acc, int N)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/// Calculem la interaccio de cada particula amb la NP
	while (i < N) 
	{
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		/// Type of the current particle
		int typei = type[i];
		
		/// Properties of the given type
		float EpsNPi = dev_EpsNP[typei];
		float Ri = dev_R[typei];
		float EpsEDLi = dev_EpsEDL[typei];
		
		/// Read partice positions. Coalesced read
		float4 Coords = Coord[i];
		
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		/// Squared Distance to te NP
		float r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;
		/// Distance between the center of a protein and the NP
		float r1 = sqrt(r2);  
		/// Distance between the center of a protein and the surface of the NP
		float r = r1 - dev_R_NP;
		/// Distance between the surface of a protein and the surface of the NP
		float d = r - Ri;
		
		/// Rescale positions to shift the interaction potential
		Coords *= r/r1;
		/// Squared distance of a protein to the NP's surface
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;
		
		
		/// DLVO VdW dispersion forces
		float pair_force = -EpsNPi/6.f * dev_R_NP*Ri / (dev_R_NP+Ri) / (d*d);
		
		/// DLVO Born repulsion
		pair_force += 7.f*6.f*EpsNPi * dev_R_NP*Ri / (dev_R_NP+Ri) * pow(0.5f, 6.f) / (7560.f * pow(d, 8.f));
		
		/// DLVO Electrical Double-layer forces
		pair_force += EpsEDLi * dev_kappa * exp(- dev_kappa * d);
		
		
		/// Update particle accelerations
		
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		/// Coalesced write
		Acc[i] += pair_force * Coords / sqrt(r2);
		/// Compute the resulting acceleration: a = F / m
		Acc[i] /= dev_M[typei];      
		
		/// Next particle
		i += blockDim.x*gridDim.x;   
	}
}


__global__ void gpu_forceNP_Equil_kernel (char * type, float4 * Coord, float4 * Acc, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/// Calculem la interaccio de cada particula amb la NP
	while (i < N) 
	{
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		/// Type of the current particle
		int typei = type[i];  
		
		/// Properties of the given type
		float EpsNPi = dev_EpsNP[typei];
		float Ri = dev_R[typei];
		
		/// Read partice positions. Coalesced read
		float4 Coords = Coord[i];  
		
		/// Coordinates relative to the center of the NP
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		/// Squared distance to te NP
		float r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;  
		
		/// Distance between the center of a protein and the center of the NP
		float r = sqrtf(r2);       
		/// Distance between the center of a protein and the surface of the NP
		r -= dev_R_NP;       
		
		/// Rescale the vector distance to shift the interaction potential
		Coords *= r/sqrtf(r2);
		
		/// Squared distance of a protein to the NP's surface
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;
		
		float pair_force = gpu_calcPNPForceEquil(EpsNPi, Ri*Ri, r2);
		
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		/// Coalesced write
		Acc[i] += pair_force * Coords;
		/// Compute the resulting acceleration: a = F / m
		Acc[i] /= dev_M[typei];      
		
		/// Next particle
		i += blockDim.x*gridDim.x;
	}
}


/// NOT IN USE
void verlet_calcForce (bool equil) 
{
	/** This calculates the resultant force over
	   each particle. Is also part of the calculation
	   of the Pressure */
	
	/** If we have N = 2048 particles, and we use nthreads = 128
	 *  we can launch exactly 2048 / 128 = 16 blocks that can fully synchronize.
	 *  Otherwise, if for example N = 2049, we need 128 * 16 + 1 threads, so 17 blocks
	 *  the last block will only have 1 active thread, and cannot synchronize. */ 
	
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads; /// Synchronous blocks
	
	/// Check the force over each particle due to pair interactions with neighbour particles. Omit the NP
	if(equil) {
		gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	else {
		gpu_forceThreadPerAtomTabulatedSync_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_tableF_att_symm, dev_tableF_att_noSymm, dev_nlist, dev_vlist, N);
		if(debug)
			CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
				CudaCheckError();
		}
	}
	
	if(calc_thermo) {
		CudaSafeCall(cudaMemset(dev_Epot_NP, 0, sizeof(float)));
		CudaSafeCall(cudaMemset(dev_P_NP,    0, sizeof(float)));
	}
	//force_wall1_kernel <<<(N+191)/192,192>>> (/*dev_z,*/ dev_R, dev_M, dev_EpsNP, dev_az, N, Lz, epsNP, T0);	
	//force_wall2_kernel <<<(N+191)/192,192>>> (/*dev_z,*/ dev_R, dev_M, dev_EpsNP, dev_az, N, Lz, epsNP, T0);
	
	/// Add to the forces the contribution of the NP in a separate calculation
	if(equil)
		gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, N);
	else
		gpu_forceNP_DLVO_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, N);
	
	if(debug)
		CudaCheckError();
	
	/// Divide force over mass to obtain accelerations
	/// Now, It is done at the end of force_NP_kernel! 
	//divide_acc_mass_kernel <<<(N+191)/192,192>>> (/*dev_ax, dev_ay, dev_az,*/ dev_Acc/*,  dev_type, N*/);
}

void verlet_calcForceEquil (void) 
{
	/** This calculates the resultant force over
	   each particle. Is also part of the calculation
	   of the Pressure */
	
	/** If we have N = 2048 particles, and we use nthreads = 128
	 *  we can launch exactly 2048 / 128 = 16 blocks that can fully synchronize.
	 *  Otherwise, if for example N = 2049, we need 128 * 16 + 1 threads, so 17 blocks
	 *  the last block will only have 1 active thread, and cannot synchronize. */ 
	
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads; /// Synchronous blocks
	
	/// Check the force over each particle due to pair interactions with neighbour particles. Omit the NP
	gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N);
	if(debug)
		CudaCheckError();
	
	if(nblocks*nthreads < N) {      /// Asynchronous block
		gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N, nblocks*nthreads);
		if(debug)
		CudaCheckError();
	}
	
	if(calc_thermo) {
		CudaSafeCall(cudaMemset(dev_Epot_NP, 0, sizeof(float)));
		CudaSafeCall(cudaMemset(dev_P_NP,    0, sizeof(float)));
	}
	
	/// Add to the forces the contribution of the NP in a separate calculation
	gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, N);
	
	if(debug)
		CudaCheckError();
	
	/// Divide force over mass to obtain accelerations
	/// Now, It is done at the end of force_NP_kernel! 
	//divide_acc_mass_kernel <<<(N+191)/192,192>>> (/*dev_ax, dev_ay, dev_az,*/ dev_Acc/*,  dev_type, N*/);
}


void verlet_calcForceLangevin (bool equil) 
{
	/** This calculates the resultant force over
	   each particle. Is also part of the calculation
	   of the Pressure */
	
	/** If we have N = 2048 particles, and we use nthreads = 128
	 *  we can launch exactly 2048 / 128 = 16 blocks that can fully synchronize.
	 *  Otherwise, if for example N = 2049, we need 128 * 16 + 1 threads, so 17 blocks
	 *  the last block will only have 1 active thread, and cannot synchronize. */ 
	
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads; /// Synchronous blocks
	
	/// Check the force over each particle due to pair interactions with neighbour particles. Omit the NP
	if(equil) {
		gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	else {
		gpu_forceThreadPerAtomTabulatedSync_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_tableF_att_symm, dev_tableF_att_noSymm, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_type, dev_Coord, dev_Acc, dev_tableF_rep, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	
	/// Add to the forces the contribution of the NP in a separate calculation
	if(equil)
		gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, N);
	else 
		gpu_forceNP_DLVO_kernel <<<nblocks,nthreads>>> (dev_type, dev_Coord, dev_Acc, N);
	
	
	gpu_RNG_generate <<<nblocks,nthreads>>> ( devStates, dev_Rnd, dev_type, N); 
	
	gpu_addRandomForces_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_Rnd, N);
	if(debug)
		CudaCheckError();
	
	/// Divide force over mass to obtain accelerations
	/// Now, It is done at the end of force_NP_kernel! 
	//divide_acc_mass_kernel <<<(N+191)/192,192>>> (/*dev_ax, dev_ay, dev_az,*/ dev_Acc/*,  dev_type, N*/);
}

void util_calcPPForceTable (void) 
{
	ntable = (int)(rc*rc * 100 + 1);
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_ntable, (&ntable), sizeof(int)));
	
	float * tableU_rep =        (float *)malloc(ntypes*ntypes*ntable*sizeof(float));
	tableF_rep =        (float *)malloc(ntypes*ntypes*ntable*sizeof(float));
	tableF_att_symm =   (float *)malloc(ntypes*ntypes*ntable*sizeof(float));
	tableF_att_noSymm = (float *)malloc(ntypes*ntypes*ntable*sizeof(float));
	
	CudaSafeCall(cudaMalloc( (void**)&dev_tableF_rep,        ntypes*ntypes*ntable*sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_tableF_att_symm,   ntypes*ntypes*ntable*sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_tableF_att_noSymm, ntypes*ntypes*ntable*sizeof(float) ));
	
	for(int t1 = 0; t1 < ntypes; ++t1)
	  for(int t2 = 0; t2 < ntypes; ++t2)
	  {
		float w   = (R[t1]+R[t2])*0.25f;
		float eps = -sqrt(EpsProt[t1]*EpsProt[t2]);
		float d   = (R[t1] + R[t2]);
		float D   = Rs[t1] + Rs[t2];
		
		if(epsDiag == 0) {
			if(t1 != t2)
				eps = 0.;
		}
		
		for(double r2 = 0; r2 < rc*rc; r2+=1./100)
		{
			int index = (int)(r2*100+0.5) + ntable*(t2 + ntypes*t1);
			
			double r = sqrt(r2);
			
			if(r > d*0.5)
			{
				/// Correct barrier for correct diffusion: 2.45
				tableU_rep[index] = pow(d/r,24) + 2. / (1 + exp(30 * (r - D)/d));
				
				tableF_rep[index] = 24 * pow(d/r,24) / (r*r) + 2. * 30. / (r * d * pow(2*cosh(0.5 * 30 * (r - D)/d), 2));
				
				tableF_att_symm[index] = -eps * (r - 1.1*d)/(w*w) * exp(-pow(r - 1.1*d,2)/(2*w*w)) / r;
				
				tableF_att_noSymm[index] = -eps * exp(-pow(r - 1.1*d,2)/(2*w*w));
			}
			else
				tableU_rep[index] = tableF_rep[index] = tableF_att_symm[index] = tableF_att_noSymm[index] = 0.;
		}
		
	  }
	  
	  FILE * myFile, * myFile2, * myFile3, * myFile4;
	  myFile = fopen("forces.dat","w");
	  myFile2 = fopen("potentials.dat","w");
	  myFile3 = fopen("forces-NP.dat","w");
	  myFile4 = fopen("potentials-NP.dat","w");
	  
	  printf("Computing potential tables\n");
	  fflush(stdout);
	  
	  for(double r2 = 0; r2 < rc*rc; r2+=1./100) {
		
		fprintf(myFile, "%f\t", sqrt(r2));
		fprintf(myFile2, "%f\t", sqrt(r2));
		fprintf(myFile3, "%f\t", sqrt(r2));
		fprintf(myFile4, "%f\t", sqrt(r2));
		
		for(int t1 = 0; t1 < ntypes; ++t1) {
			for(int t2 = 0; t2 < ntypes; ++t2) {
				int index = (int)(r2*100+0.5) + ntable*(t2 + ntypes*t1);
				
				fprintf(myFile, "%f\t", tableF_rep[index]);
				fprintf(myFile2, "%f\t", tableU_rep[index]);
			}
			
			
			/// DLVO VdW dispersion forces
			
			float d = sqrt(r2);    /// Distance between the surface of a protein and the surface of the NP
			
			
			float pair_force = -EpsNP[t1]/6. * R_NP*R[t1] / (R_NP+R[t1]) / (d*d);
			float pair_potential = -EpsNP[t1]/6. * R_NP*R[t1] / (R_NP+R[t1]) / d;
			
			/// DLVO Born repulsion
			
			pair_force += 7.*6.*EpsNP[t1] * R_NP*R[t1] / (R_NP+R[t1]) * pow(0.5, 6.) / (7560. * pow(d, 8.));
			pair_potential += 6.*EpsNP[t1] * R_NP*R[t1] / (R_NP+R[t1]) * pow(0.5f, 6.f) / (7560. * pow(d, 7.));
			
			/// DLVO Electrical Double-layer forces
			pair_force += EpsEDL[t1] * kappa * exp(- kappa * d);
			pair_potential += EpsEDL[t1] * exp(- kappa * d);
			
			fprintf(myFile3, "%f\t", pair_force);
			fprintf(myFile4, "%f\t", pair_potential);
		}
		
		fprintf(myFile, "\n");
		fprintf(myFile2, "\n");
		fprintf(myFile3, "\n");
		fprintf(myFile4, "\n");
	}
	
	printf("End of Computing potential tables\n");
	fflush(stdout);
	
	fclose(myFile);
	fclose(myFile2);
	fclose(myFile3);
	fclose(myFile4);
}


void initialize (int argc, char * argv[]) 
{
	///seed = time(NULL) or seed = 1234 for debugging
	srand(1234);                           /// Random seed
	
	//////////////// Initial configuration load ////////////////////////
	if(verbose)
		printf("Starting conf read...\n");
	
	char filename[100];
	
	/// Default configuration file
	sprintf(filename, "setup.dat");
	
	/// Different if specified as command-line argument
	if(argc > 1) {
		sprintf(filename, argv[1]);
		printf("Configuration file is: %s\n", filename);
	}
	
	init_config(filename, sizeof(filename));
	
	if(verbose) {
		printf("Init config read\n");
		fflush(stdout);
	}
	
	if(restart)                           /// Check if we are starting from a previous run
		util_loadConfig();
	
	if(verbose) {
		printf("Configuration read\n");
		printf("Number of particles: %d\n", N);
		fflush(stdout);
	}
	
//	util_calcPPpotentialTable();                 /// Compute the potential tables
	util_calcPPForceTable();                 /// Compute the force tables
	
	T = thermo_temperature();         /// Compute the temperature
	if(verbose) {
		printf("Temperature: %lf\n\n",T);
		fflush(stdout);
	}
	
	
	langPrefactor = (float *)malloc(ntypes*sizeof(float));
	
	langPrefactor[0] = sqrt(2.*T0*Gamma1/dt);
	langPrefactor[1] = sqrt(2.*T0*Gamma2/dt);
	langPrefactor[2] = sqrt(2.*T0*Gamma3/dt);

	
	printf("lang prefactor 1 %f\n", langPrefactor[0]);
	printf("lang prefactor 2 %f\n", langPrefactor[1]);
	printf("lang prefactor 3 %f\n", langPrefactor[2]);
	
	////////////// Copy system state from CPU to GPU ///////////////////
	/// Dynamic variables
	CudaSafeCall(cudaMemcpy( dev_Coord, Coord, N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_Vel,   Vel,   N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_Acc,   Acc,   N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_type,  type,  N*sizeof(char),   cudaMemcpyHostToDevice ));
	
	CudaSafeCall(cudaMemcpy(dev_gamma, Gamma, N*sizeof(float), cudaMemcpyHostToDevice));

	/// Set constant parameters in GPU
	CudaSafeCall(cudaMemcpyToSymbol(dev_N, &N, sizeof(int)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_L,  (&L),  sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_hL, (&hL), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_V, (&V), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_kappa, (&kappa), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_K, (&K), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_rc2_NP, (&rc2_NP), sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_rv2,    (&rv2),    sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_skin2,  (&skin2),  sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_vec_rc2,  vec_rc2,    ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_vec_rv2,  vec_rv2,    ntypes*sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_dt,  (&dt),  sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_hdt, (&hdt), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_langPrefactor, langPrefactor, ntypes*sizeof(float)));
	
	/// Particle properties
	CudaSafeCall(cudaMemcpyToSymbol(dev_M,     M,     ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_sqrM,  sqrM,  ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_R,     R,     ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_Rs,    Rs,    ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_EpsNP, EpsNP, ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_EpsEDL, EpsEDL, ntypes*sizeof(float)));
	
	
	/// NP properties
	CudaSafeCall(cudaMemcpyToSymbol(dev_x_NP, (&x_NP), sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_y_NP, (&y_NP), sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_z_NP, (&z_NP), sizeof(float)));
	
	float4 xyz_NP = make_float4(x_NP,y_NP,z_NP,0.);
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_xyzNP, (&xyz_NP), sizeof(float4)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_R_NP,  (&R_NP),   sizeof(float)));
	
	/// Potential tables
//	CudaSafeCall(cudaMemcpy(dev_tableU_rep, tableU_rep, ntable*sizeof(float), cudaMemcpyHostToDevice));
//	CudaSafeCall(cudaMemcpy(dev_tableU_att, tableU_att, ntable*sizeof(float), cudaMemcpyHostToDevice));
	
	/// Force tables
	CudaSafeCall(cudaMemcpy(dev_tableF_rep,        tableF_rep,        ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_tableF_att_symm,   tableF_att_symm,   ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_tableF_att_noSymm, tableF_att_noSymm, ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));

	////////////////////////// Initialization //////////////////////////
	init_rdf() ;
	
	/// Compute the Verlet List at the beginning 
	util_calcVerletList();
	
	/// Then compute the forces, before first Coord and Vel update
	verlet_calcForceLangevin(TRUE);
}

///
__global__ void gpu_updateVelocities_kernel (
	float4 * Vel, 
	float4 * Acc, 
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] += Acc[tid] * dev_hdt;
		
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_updateVelocitiesLangevin_kernel (
	float4 * Vel, 
	float4 * Acc, 
	float * gamma,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] += Acc[tid] * dev_hdt;
		Vel[tid] /= 1.f + gamma[tid] * dev_hdt;
		
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_updateVelocitiesLangevinRelax_kernel (
	float4 * Vel, 
	float4 * Acc, 
	float * dev_gamma,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] += Acc[tid] * dev_hdt;
		Vel[tid] /= 1.f + dev_gamma[tid] * dev_hdt;
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_rescaleVelocities_kernel (
	float4 * Vel, 
	float scale, 
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] *= scale;
		
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_updateVelocitiesPositions_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float4 coord, coordIni, coordFin;
	float4 vel;
	
	int isXiniInside = 1, isYiniInside = 1, isZiniInside = 1;
	int isXfinInside = 1, isYfinInside = 1, isZfinInside = 1;
	int isXcrossingIn = 0, isYcrossingIn = 0, isZcrossingIn = 0;
	int isXcrossingOut = 0, isYcrossingOut = 0, isZcrossingOut = 0;
	
	int threadIdMax = N/blockDim.x * blockDim.x;
	
	float dev_LbuffMin = dev_L*0.1;
	float dev_LbuffMax = dev_L*0.9;
	
	while (tid < N) 
	{
		if(tid < threadIdMax)
			__syncthreads();
		
		vel = Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		Vel[tid] = vel;
		
		coordIni = Coord[tid];                 /// Coalesced read
		coordFin = coordIni + vel * dev_dt;              /// Advance coordinates
		
		coord = coordFin;
		
		if(coordIni.x < dev_LbuffMin)
			isXiniInside = 0;
		if(coordIni.x > dev_LbuffMax)
			isXiniInside = 0;
		
		if(coordIni.y < dev_LbuffMin)
			isYiniInside = 0;
		if(coordIni.y > dev_LbuffMax)
			isYiniInside = 0;
		
		if(coordIni.z < dev_LbuffMin)
			isZiniInside = 0;
		if(coordIni.z > dev_LbuffMax)
			isZiniInside = 0;
		
		
		if(coordFin.x < dev_LbuffMin)
			isXfinInside = 0;
		if(coordFin.x > dev_LbuffMax)
			isXfinInside = 0;
		
		if(coordFin.y < dev_LbuffMin)
			isYiniInside = 0;
		if(coordFin.y > dev_LbuffMax)
			isYfinInside = 0;
		
		if(coordFin.z < dev_LbuffMin)
			isZfinInside = 0;
		if(coordFin.z > dev_LbuffMax)
			isZfinInside = 0;
		
		
		if(isXiniInside && !isXfinInside)
			isXcrossingOut = 1;
		if(isYiniInside && !isYfinInside)
			isYcrossingOut = 1;
		if(isZiniInside && !isZfinInside)
			isZcrossingOut = 1;
		
		if(!isXiniInside && isXfinInside)
			isXcrossingIn = 1;
		if(!isYiniInside && isYfinInside)
			isYcrossingIn = 1;
		if(!isZiniInside && isZfinInside)
			isZcrossingIn = 1;
		
		if(isXcrossingOut || isYcrossingOut || isZcrossingOut)
			coord = coordIni - vel * dev_dt; /// turn back !
		
		if(isXcrossingIn || isYcrossingIn || isZcrossingIn)
			coord = coordIni - vel * dev_dt; /// turn back !
		
		/// Periodic boundary conditions
		if(coord.x < 0.f)
			coord.x += dev_L;
		else if(coord.x > dev_L)
			coord.x -= dev_L;
		
		if(coord.y < 0.f)
			coord.y += dev_L;
		else if(coord.y > dev_L)
			coord.y -= dev_L;
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f)
			coord.z += dev_L;
		else if(coord.z > dev_L)
			coord.z -= dev_L;
		
		if(tid < threadIdMax)
			__syncthreads();
		
		Coord[tid] = coord;                     /// Assign new position
		
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_updateVelocitiesPositionsLangevin_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int * isOutsideClosed,
	float * dev_gamma,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	int isXiniInside, isYiniInside, isZiniInside;
	int isXfinInside, isYfinInside, isZfinInside;
	
	while (tid < N) 
	{
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		float4 vel = (1.f - dev_gamma[tid] * dev_hdt) * Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		float4 coordIni = Coord[tid];                 /// Coalesced read
		float4 coord = coordIni + vel * dev_dt;              /// Advance coordinates
		
		/// check if the proteins can move from the buffer to the reaction region
		if(isOutsideClosed[tid]) {
			
			isXiniInside = 1;
			isYiniInside = 1;
			isZiniInside = 1;
			
			isXfinInside = 1;
			isYfinInside = 1;
			isZfinInside = 1;
			
			if(coordIni.x < dev_L*0.1f)
				isXiniInside = 0;
			else if(coordIni.x > dev_L*0.9f)
				isXiniInside = 0;
			
			if(coordIni.y < dev_L*0.1f)
				isYiniInside = 0;
			else if(coordIni.y > dev_L*0.9f)
				isYiniInside = 0;
			
			if(coordIni.z < dev_L*0.1f)
				isZiniInside = 0;
			else if(coordIni.z > dev_L*0.9f)
				isZiniInside = 0;
			
			
			if(coord.x < dev_L*0.1f)
				isXfinInside = 0;
			else if(coord.x > dev_L*0.9f)
				isXfinInside = 0;
			
			if(coord.y < dev_L*0.1f)
				isYfinInside = 0;
			else if(coord.y > dev_L*0.9f)
				isYfinInside = 0;
			
			if(coord.z < dev_L*0.1f)
				isZfinInside = 0;
			else if(coord.z > dev_L*0.9f)
				isZfinInside = 0;
			
			/// check if protein is finally inside the reaction region
			if(isXfinInside && isYfinInside && isZfinInside) {
				if(!isXiniInside) {
					coord.x = coordIni.x - vel.x * dev_dt;
					vel.x *= -1;
				}
				if(!isYiniInside) {
					coord.y = coordIni.y - vel.y * dev_dt;
					vel.y *= -1;
				}
				if(!isZiniInside) {
					coord.z = coordIni.z - vel.z * dev_dt;
					vel.z *= -1;
				}
			}
		}
		
		/// Periodic boundary conditions
		if(coord.x < 0.f)
			coord.x += dev_L;
		
		else if(coord.x > dev_L) 
			coord.x -= dev_L;
		
		
		if(coord.y < 0.f)
			coord.y += dev_L;
		
		else if(coord.y > dev_L) 
			coord.y -= dev_L;
		
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f)
			coord.z += dev_L;
		
		else if(coord.z > dev_L) 
			coord.z -= dev_L;
		
		
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		Coord[tid] = coord;                     /// Assign new position
		
		Vel[tid] = vel;
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_updateVelocitiesPositionsLangevinEquil_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int * isOutsideClosed,
	float * dev_gamma,
	int N,
	unsigned int * newlist) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	int isXiniInside, isYiniInside, isZiniInside;
	int isXfinInside, isYfinInside, isZfinInside;
	
	unsigned char check = 1;
	
	if(threadIdx.x == 0)
		(*newlist) = 0;
	
	while (tid < N) 
	{
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		float4 vel = (1.f - dev_gamma[tid] * dev_hdt) * Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		float4 coordIni = Coord[tid];                 /// Coalesced read
		float4 coord = coordIni + vel * dev_dt;              /// Advance coordinates
		
		/// check if the proteins can move from the buffer to the reaction region
		if(isOutsideClosed[tid]) {
			
			isXiniInside = 1;
			isYiniInside = 1;
			isZiniInside = 1;
			
			isXfinInside = 1;
			isYfinInside = 1;
			isZfinInside = 1;
			
			if(coordIni.x < dev_L*0.1f)
				isXiniInside = 0;
			else if(coordIni.x > dev_L*0.9f)
				isXiniInside = 0;
			
			if(coordIni.y < dev_L*0.1f)
				isYiniInside = 0;
			else if(coordIni.y > dev_L*0.9f)
				isYiniInside = 0;
			
			if(coordIni.z < dev_L*0.1f)
				isZiniInside = 0;
			else if(coordIni.z > dev_L*0.9f)
				isZiniInside = 0;
			
			
			if(coord.x < dev_L*0.1f)
				isXfinInside = 0;
			else if(coord.x > dev_L*0.9f)
				isXfinInside = 0;
			
			if(coord.y < dev_L*0.1f)
				isYfinInside = 0;
			else if(coord.y > dev_L*0.9f)
				isYfinInside = 0;
			
			if(coord.z < dev_L*0.1f)
				isZfinInside = 0;
			else if(coord.z > dev_L*0.9f)
				isZfinInside = 0;
			
			/// check if protein is finally inside the reaction region
			if(isXfinInside && isYfinInside && isZfinInside) {
				if(!isXiniInside) {
					coord.x = coordIni.x - vel.x * dev_dt;
					vel.x *= -1;
				}
				if(!isYiniInside) {
					coord.y = coordIni.y - vel.y * dev_dt;
					vel.y *= -1;
				}
				if(!isZiniInside) {
					coord.z = coordIni.z - vel.z * dev_dt;
					vel.z *= -1;
				}
			}
		}
		
		/// Periodic boundary conditions
		if(coord.x < 0.f)
			coord.x += dev_L;
		
		else if(coord.x > dev_L) 
			coord.x -= dev_L;
		
		
		if(coord.y < 0.f)
			coord.y += dev_L;
		
		else if(coord.y > dev_L) 
			coord.y -= dev_L;
		
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f)
			coord.z += dev_L;
		
		else if(coord.z > dev_L) 
			coord.z -= dev_L;
		
		
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		Coord[tid] = coord;                     /// Assign new position
		
		Vel[tid] = vel;
		
		if(check) /// Check the distance to the original position of last verlet list
		{
			coord -= CoordVerlet[tid];
			
			if(coord.x > dev_hL)
				coord.x -= dev_L;
			else if(coord.x < -dev_hL)
				coord.x += dev_L;
			
			if(coord.y > dev_hL)
				coord.y -= dev_L;
			else if(coord.y < -dev_hL)
				coord.y += dev_L;
				
			/// NO NEED if Wall
			if(coord.z > dev_hL)
				coord.z -= dev_L;
			else if(coord.z < -dev_hL)
				coord.z += dev_L;
			
			/// flag to check if we need to recompute the verlet list
			if(coord.x*coord.x + coord.y*coord.y + coord.z*coord.z > dev_skin2) 
			{
				atomicInc(newlist,1);
				check = 0;
			}
		}
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_updateVelocitiesPositionsLangevinMSD_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int4 * image,
	int * isOutsideClosed,
	float * dev_gamma,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	int isXiniInside, isYiniInside, isZiniInside;
	int isXfinInside, isYfinInside, isZfinInside;
	
	while (tid < N) 
	{
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		float4 vel = (1.f - dev_gamma[tid] * dev_hdt) * Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		
		
		float4 coordIni = Coord[tid];                 /// Coalesced read
		float4 coord = coordIni + vel * dev_dt;              /// Advance coordinates
		
		/// check if the proteins can move from the buffer to the reaction region
		if(isOutsideClosed[tid]) {
			
			isXiniInside = 1;
			isYiniInside = 1;
			isZiniInside = 1;
			
			isXfinInside = 1;
			isYfinInside = 1;
			isZfinInside = 1;
			
			if(coordIni.x < dev_L*0.1f)
				isXiniInside = 0;
			else if(coordIni.x > dev_L*0.9f)
				isXiniInside = 0;
			
			if(coordIni.y < dev_L*0.1f)
				isYiniInside = 0;
			else if(coordIni.y > dev_L*0.9f)
				isYiniInside = 0;
			
			if(coordIni.z < dev_L*0.1f)
				isZiniInside = 0;
			else if(coordIni.z > dev_L*0.9f)
				isZiniInside = 0;
			
			
			if(coord.x < dev_L*0.1f)
				isXfinInside = 0;
			else if(coord.x > dev_L*0.9f)
				isXfinInside = 0;
			
			if(coord.y < dev_L*0.1f)
				isYfinInside = 0;
			else if(coord.y > dev_L*0.9f)
				isYfinInside = 0;
			
			if(coord.z < dev_L*0.1f)
				isZfinInside = 0;
			else if(coord.z > dev_L*0.9f)
				isZfinInside = 0;
			
			/// check if protein is finally inside the reaction region
			if(isXfinInside && isYfinInside && isZfinInside) {
				if(!isXiniInside) {
					coord.x = coordIni.x - vel.x * dev_dt;
					vel.x *= -1;
				}
				if(!isYiniInside) {
					coord.y = coordIni.y - vel.y * dev_dt;
					vel.y *= -1;
				}
				if(!isZiniInside) {
					coord.z = coordIni.z - vel.z * dev_dt;
					vel.z *= -1;
				}
			}
		}
		
		/// Periodic boundary conditions
		if(coord.x < 0.f){
			coord.x += dev_L;
			image[tid].x -= 1;
		}
		else if(coord.x > dev_L) {
			coord.x -= dev_L;
			image[tid].x += 1;
		}
		
		if(coord.y < 0.f){
			coord.y += dev_L;
			image[tid].y -= 1;
		}
		else if(coord.y > dev_L) {
			coord.y -= dev_L;
			image[tid].y += 1;
		}
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f){
			coord.z += dev_L;
			image[tid].z -= 1;
		}
		else if(coord.z > dev_L) {
			coord.z -= dev_L;
			image[tid].z += 1;
		}
		
		///if(tid < threadIdMax)
		if(tid < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		Coord[tid] = coord;                     /// Assign new position
		
		Vel[tid] = vel;
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_updateVelocitiesPositionsLangevinRelax_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int * isOutsideClosed,
	float * dev_gamma,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float4 coord, coordIni;
	float4 vel;
	
	int isXiniInside, isYiniInside, isZiniInside;
	int isXfinInside, isYfinInside, isZfinInside;
	
	int threadIdMax = N/blockDim.x * blockDim.x;
	
	while (tid < N) 
	{
		if(tid < threadIdMax)
			__syncthreads();
		
		vel = (1.f - dev_gamma[tid] * dev_hdt) * Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		coordIni = Coord[tid];                 /// Coalesced read
		coord = coordIni + vel * dev_dt;              /// Advance coordinates
		
		/// check if the proteins can move from the buffer to the reaction region
		if(isOutsideClosed[tid]) {
			
			isXiniInside = 1;
			isYiniInside = 1;
			isZiniInside = 1;
			
			isXfinInside = 1;
			isYfinInside = 1;
			isZfinInside = 1;
			
			if(coordIni.x < dev_L*0.1f)
				isXiniInside = 0;
			else if(coordIni.x > dev_L*0.9f)
				isXiniInside = 0;
			
			if(coordIni.y < dev_L*0.1f)
				isYiniInside = 0;
			else if(coordIni.y > dev_L*0.9f)
				isYiniInside = 0;
			
			if(coordIni.z < dev_L*0.1f)
				isZiniInside = 0;
			else if(coordIni.z > dev_L*0.9f)
				isZiniInside = 0;
			
			
			if(coord.x < dev_L*0.1f)
				isXfinInside = 0;
			else if(coord.x > dev_L*0.9f)
				isXfinInside = 0;
			
			if(coord.y < dev_L*0.1f)
				isYfinInside = 0;
			else if(coord.y > dev_L*0.9f)
				isYfinInside = 0;
			
			if(coord.z < dev_L*0.1f)
				isZfinInside = 0;
			else if(coord.z > dev_L*0.9f)
				isZfinInside = 0;
			
			/// check if protein is finally inside the reaction region
			if(isXfinInside && isYfinInside && isZfinInside) {
				if(!isXiniInside) {
					coord.x = coordIni.x - vel.x * dev_dt;
					vel.x *= -1;
				}
				if(!isYiniInside) {
					coord.y = coordIni.y - vel.y * dev_dt;
					vel.y *= -1;
				}
				if(!isZiniInside) {
					coord.z = coordIni.z - vel.z * dev_dt;
					vel.z *= -1;
				}
			}
		}
		
		/// Periodic boundary conditions
		if(coord.x < 0.f){
			coord.x += dev_L;
		}
		else if(coord.x > dev_L) {
			coord.x -= dev_L;
		}
		
		if(coord.y < 0.f){
			coord.y += dev_L;
		}
		else if(coord.y > dev_L) {
			coord.y -= dev_L;
		}
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f){
			coord.z += dev_L;
		}
		else if(coord.z > dev_L) {
			coord.z -= dev_L;
		}
		
		if(tid < threadIdMax)
			__syncthreads();
		
		Coord[tid] = coord;                     /// Assign new position
		
		Vel[tid] = vel;
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_updateVelocitiesPositionsEquil_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int * isOutsideClosed,
	int N, 
	unsigned int * newlist) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float4 coord, coordIni;
	float4 vel;
	unsigned char check = 1;
	
	int isXiniInside = 1, isYiniInside = 1, isZiniInside = 1;
	int isXfinInside = 1, isYfinInside = 1, isZfinInside = 1;
	
	int threadIdMax = N/blockDim.x * blockDim.x;
	
	if(threadIdx.x == 0)
		(*newlist) = 0;
	
	while (tid < N) 
	{
		if(tid < threadIdMax)
			__syncthreads();
		
		vel = Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		
		
		coordIni = Coord[tid];                 /// Coalesced read
		coord = coordIni + vel * dev_dt;              /// Advance coordinates
		
		/// check if the proteins can move from the buffer to the reaction region
		if(isOutsideClosed[tid]) {
			
			isXiniInside = 1;
			isYiniInside = 1;
			isZiniInside = 1;
			
			isXfinInside = 1;
			isYfinInside = 1;
			isZfinInside = 1;
			
			if(coordIni.x < dev_L*0.1f)
				isXiniInside = 0;
			else if(coordIni.x > dev_L*0.9f)
				isXiniInside = 0;
			
			if(coordIni.y < dev_L*0.1f)
				isYiniInside = 0;
			else if(coordIni.y > dev_L*0.9f)
				isYiniInside = 0;
			
			if(coordIni.z < dev_L*0.1f)
				isZiniInside = 0;
			else if(coordIni.z > dev_L*0.9f)
				isZiniInside = 0;
			
			
			if(coord.x < dev_L*0.1f)
				isXfinInside = 0;
			else if(coord.x > dev_L*0.9f)
				isXfinInside = 0;
			
			if(coord.y < dev_L*0.1f)
				isYfinInside = 0;
			else if(coord.y > dev_L*0.9f)
				isYfinInside = 0;
			
			if(coord.z < dev_L*0.1f)
				isZfinInside = 0;
			else if(coord.z > dev_L*0.9f)
				isZfinInside = 0;
			
			/// check if protein is finally inside the reaction region
			if(isXfinInside && isYfinInside && isZfinInside) {
				if(!isXiniInside) {
					coord.x = coordIni.x - vel.x * dev_dt;
					vel.x *= -1;
				}
				if(!isYiniInside) {
					coord.y = coordIni.y - vel.y * dev_dt;
					vel.y *= -1;
				}
				if(!isZiniInside) {
					coord.z = coordIni.z - vel.z * dev_dt;
					vel.z *= -1;
				}
			}
		}
		
		/// Periodic boundary conditions
		if(coord.x < 0.f)
			coord.x += dev_L;
		else if(coord.x > dev_L)
			coord.x -= dev_L;
		
		if(coord.y < 0.f)
			coord.y += dev_L;
		else if(coord.y > dev_L)
			coord.y -= dev_L;
		
		/// NO NEED if Wall is used
		if(coord.z < 0.f)
			coord.z += dev_L;
		else if(coord.z > dev_L)
			coord.z -= dev_L;
		
		if(tid < threadIdMax)
			__syncthreads();
		
		Coord[tid] = coord;  /// Assign new position
		
		Vel[tid] = vel;
		
		if(check) /// Check the distance to the original position of last verlet list
		{
			coord -= CoordVerlet[tid];
			
			if(coord.x > dev_hL)
				coord.x -= dev_L;
			else if(coord.x < -dev_hL)
				coord.x += dev_L;
			
			if(coord.y > dev_hL)
				coord.y -= dev_L;
			else if(coord.y < -dev_hL)
				coord.y += dev_L;
				
			/// NO NEED if Wall
			if(coord.z > dev_hL)
				coord.z -= dev_L;
			else if(coord.z < -dev_hL)
				coord.z += dev_L;
			
			/// flag to check if we need to recompute the verlet list
			if(coord.x*coord.x + coord.y*coord.y + coord.z*coord.z > dev_skin2) 
			{
				atomicInc(newlist,1);
				check = 0;
			}
		}
		
		tid += blockDim.x*gridDim.x;
	}
}


/// Berendsen thermostat. Velocity-scaling
__global__ void gpu_thermostatBerendsen_kernel (
	float4 * Vel, 
	double * T, 
	float T0,
	float tau,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ float sChi;
	
	/// The first thread computes the thermostat parameter and saves it in shared mem
	if(threadIdx.x == 0)
		sChi = sqrtf((tau-dev_dt)/tau + (dev_dt*T0)/( tau * (*T) ));
	
	__syncthreads();
	
	float Chi = sChi;
	
	while (tid < N) {
		Vel[tid] *= Chi;              /// Coalesced writes
		
		tid += blockDim.x*gridDim.x;
	}
}


__global__ void gpu_partialTemp_kernel (
	char * type, 
	float4 * Vel, 
	int N, 
	double * partialT) 
{
	extern __shared__ double v2_cache[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	v2_cache[threadIdx.x] = 0.f;    /// Initialize cache
	
	__syncthreads();
	
	while(tid < N) 
	{
		float4 vel = Vel[tid];   /// Read velocity from global memory
		
		vel.w = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;       /// Compute square sum of components
		vel.w *= dev_M[type[tid]];              /// Multiply by the mass
		
		v2_cache[threadIdx.x] += vel.w;     /// Accumulate to cache
		
		tid += blockDim.x*gridDim.x;        /// Next particle
	}
	
	__syncthreads();
	
	int nTotalThreads = blockDim.x;               /// Total number of active threads
	
	/** Algoritme per calcular la reduccio 
	 *  dels valors actuals a la cache del block */
	while(nTotalThreads > 1) 
	{
		int halfPoint = (nTotalThreads >> 1);       /// divide by two, only the first half of the threads will be active.
		
		if (threadIdx.x < halfPoint) 
			v2_cache[threadIdx.x] += v2_cache[threadIdx.x + halfPoint];
		
		__syncthreads();                /// imprescindible
		
		nTotalThreads = halfPoint;      /// Reducing the binary tree size by two:
	}
	
	/// El primer thread de cada block es el k s'encarrega de fer els calculs finals
	if(threadIdx.x == 0) 
		partialT[blockIdx.x] = v2_cache[0];    /// Add squared velocities sum to partial T
}


__global__ void gpu_totalTemp_kernel (
	int N,
	double * partialT,
	double * totalT) 
{
	extern __shared__ double T_cache[];
	int tid = threadIdx.x;
	
	T_cache[tid] = partialT[tid];
	
	__syncthreads();
	
	int nTotalThreads = blockDim.x;               /// Total number of active threads
	
	/** Algoritme per calcular la reduccio 
	 *  dels valors actuals a la cache del block */
	while(nTotalThreads > 1) 
	{
		int halfPoint = (nTotalThreads >> 1);       /// divide by two, only the first half of the threads will be active.
		
		if (threadIdx.x < halfPoint) 
			T_cache[threadIdx.x] += T_cache[threadIdx.x + halfPoint];
		
		__syncthreads();                /// imprescindible
		
		nTotalThreads = halfPoint;      /// Reducing the binary tree size by two:
	}
	
	
	/// El primer thread de cada block es el k s'encarrega de fer els calculs finals
	if(threadIdx.x == 0) {
		
		double T = T_cache[0];
		
		T /= (kb * dim * N);  /// Instantaneous temperature using the Equipartition Theorem. The kinetic energy is just K = 3N/2 kT
		
		(*totalT) = T;
	}
}


void util_saveState (long int t)
{
	FILE * config;
	int i;
	
	CudaSafeCall(cudaMemcpy( Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost ));
	CudaSafeCall(cudaMemcpy( Vel,   dev_Vel,   N*sizeof(float4), cudaMemcpyDeviceToHost ));
	CudaSafeCall(cudaMemcpy( Acc,   dev_Acc,   N*sizeof(float4), cudaMemcpyDeviceToHost ));
	CudaSafeCall(cudaMemcpy( type,  dev_type,  N*sizeof(char),   cudaMemcpyDeviceToHost ));
	
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s", root, "config/lastconfig.dat");
	config = fopen(filename, "w");
	
	fprintf(config, "%lu\t%d\t%lf\t%d\n", t, updateVerletListPeriod, logscale, restartIndex);
	
	for (i = 0; i < N; ++i)
		fprintf(config, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", 
		type[i], 
		Coord[i].x, Coord[i].y,  Coord[i].z, 
		Vel[i].x,   Vel[i].y,    Vel[i].z, 
		Acc[i].x,   Acc[i].y,    Acc[i].z,
		R[type[i]], Rs[type[i]], M[type[i]],
		EpsNP[type[i]], EpsEDL[type[i]], EpsProt[type[i]]);
	
	fclose(config);
	
	/// Keep a copy of each final configuration in a separate set of files
	snprintf(filename, sizeof(filename), "%s%s%d%s", root, "config/lastconfig_run", restartIndex, ".dat");
	config = fopen(filename, "w");
	
	fprintf(config, "%lu\t%d\t%lf\t%d\n", t, updateVerletListPeriod, logscale, restartIndex);
	
	for (i = 0; i < N; ++i)
		fprintf(config, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", 
		type[i], 
		Coord[i].x, Coord[i].y,  Coord[i].z, 
		Vel[i].x,   Vel[i].y,    Vel[i].z, 
		Acc[i].x,   Acc[i].y,    Acc[i].z,
		R[type[i]], Rs[type[i]], M[type[i]],
		EpsNP[type[i]], EpsEDL[type[i]], EpsProt[type[i]]);
	
	fclose(config);
}

void util_resetThermostat (void)
{
	/// Compute the temperature of the system in two steps, first a block-wide reduction + thread-wide reduction
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_type, dev_Vel, N, dev_partialT);
	if(debug)
		CudaCheckError();
	gpu_totalTemp_kernel   <<< 1, 64, 64*sizeof(double)>>> (N, dev_partialT, dev_T);
	if(debug)
		CudaCheckError();
	
	CudaSafeCall(cudaMemcpy (&T,dev_T,sizeof(double),cudaMemcpyDeviceToHost));
	
	chi0 = chi1 = chi2 = 0.;
}

void verlet_integrateLangevinNVT (bool computeMSD, bool updateVerletList) 
{
	/// IMPORTANT DE COMPROVAR EL NOMBRE DE THREADS I BLOCKS, I FER TESTS!!!
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads;
	
	/// First half-step
	if(computeMSD)
		gpu_updateVelocitiesPositionsLangevinMSD_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_Image, dev_isOutsideClosed, dev_gamma, N);
	else
		gpu_updateVelocitiesPositionsLangevin_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_isOutsideClosed, dev_gamma, N);
	
	if(debug)
		CudaCheckError();
	
	if(updateVerletList)
		util_calcVerletList();
	
	verlet_calcForceLangevin (FALSE);     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocitiesLangevin_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_gamma, N);
	
	if(debug)
		CudaCheckError();
}

void verlet_integrateLangevinNVTrelax (bool updateVerletList) 
{
	/// IMPORTANT DE COMPROVAR EL NOMBRE DE THREADS I BLOCKS, I FER TESTS!!!
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads;
	
	/// First half-step
	gpu_updateVelocitiesPositionsLangevinRelax_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_isOutsideClosed, dev_gamma, N);
	
	if(debug)
		CudaCheckError();
	
	if(updateVerletList)
		util_calcVerletList();
	
	verlet_calcForceLangevin (FALSE);     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocitiesLangevinRelax_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_gamma, N);
	if(debug)
		CudaCheckError();
}

void verlet_integrateLangevinNVTequil (void) 
{
	/// IMPORTANT DE COMPROVAR EL NOMBRE DE THREADS I BLOCKS, I FER TESTS!!!
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads;
	
	unsigned int newlist = 0;   /// S'inicialitza el flag que determina si cal recalcular la llista de Verlet
	
	/// First half-step
	gpu_updateVelocitiesPositionsLangevinEquil_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_isOutsideClosed, dev_gamma, N, dev_newlist);
	
	if(debug)
		CudaCheckError();
	
	/// Check if we need to recompute the Verlet list
	CudaSafeCall(cudaMemcpy( &newlist, dev_newlist, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	
	if(newlist)
		util_calcVerletList();
	
	verlet_calcForceLangevin (TRUE);     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocitiesLangevin_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_gamma, N);
	
	if(debug)
		CudaCheckError();
}

void verlet_integrateBerendsenNVT (void) 
{
	/// IMPORTANT DE COMPROVAR EL NOMBRE DE THREADS I BLOCKS, I FER TESTS!!!
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads;
	
	unsigned int newlist = 0;   /// S'inicialitza el flag que determina si cal recalcular la llista de Verlet
	
	/// First half-step
	gpu_updateVelocitiesPositionsEquil_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_isOutsideClosed, N, dev_newlist);
	if(debug)
		CudaCheckError();
	
	/// Check if we need to recompute the Verlet list
	CudaSafeCall(cudaMemcpy( &newlist, dev_newlist, sizeof(unsigned int), cudaMemcpyDeviceToHost ));
	
	if(newlist)
		util_calcVerletList();
	
	verlet_calcForceEquil();     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocities_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, N);
	if(debug)
		CudaCheckError();
	
	/// Compute the temperature of the system in two steps, first a block-wide reduction + thread-wide reduction
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_type, dev_Vel, N, dev_partialT);
	if(debug)
		CudaCheckError();
	gpu_totalTemp_kernel <<<1,64,64*sizeof(double)>>> (N, dev_partialT, dev_T);
	if(debug)
		CudaCheckError();
	
	gpu_thermostatBerendsen_kernel<<<nblocks,nthreads>>> (dev_Vel, dev_T, T0, tau, N);
	if(debug)
		CudaCheckError();
}

__global__ void gpu_reduce_kernel (int N, float * vector, float * sum) 
{
	extern __shared__ float partialSum[];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	partialSum[threadIdx.x] = 0.f;
	
	__syncthreads();
	
	while(tid < N) 
	{
		partialSum[threadIdx.x] += vector[tid];
		
		tid += blockDim.x*gridDim.x;
	}
	
	__syncthreads();
	
	int nTotalThreads = blockDim.x;	/// Total number of active threads
	
	/// Algoritme per calcular la reduccio dels valors actuals a la cache del block
	while(nTotalThreads > 1) 
	{
		int halfPoint = (nTotalThreads >> 1);	/// divide by two
		/// only the first half of the threads will be active.
		
		if (threadIdx.x < halfPoint) 
		{
				partialSum[threadIdx.x] += partialSum[threadIdx.x + halfPoint];
		}
		
		/// imprescindible per les reduccions
		__syncthreads();
		
		/// Reducing the binary tree size by two:
		nTotalThreads = halfPoint;
	}
	
	/// El primer thread del primer block es el k s'encarrega de fer els calculs finals
	if(threadIdx.x == 0) 
		(*sum) = partialSum[0];
}


void thermo_computeTemperature(void)
{
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_type, dev_Vel,N,dev_partialT);
	if(debug)
		CudaCheckError();
	gpu_totalTemp_kernel <<<1,64,64*sizeof(double)>>> (N,dev_partialT,dev_T);
	if(debug)
		CudaCheckError();
	
	CudaSafeCall(cudaMemcpy( &T, dev_T, sizeof(double), cudaMemcpyDeviceToHost ));
}


void util_removeDrift(void) /// This function computes the center of mass velocity, and sets it to zero
{
	int i;
	float Mi;
	
	/// retrieve velocities from GPU
	CudaSafeCall(cudaMemcpy( Vel, dev_Vel, N*sizeof(float4), cudaMemcpyDeviceToHost ));
	
	struct  {
		double x;
		double y;
		double z;
	} momentum;
	
	momentum.x = momentum.y = momentum.z = 0;
	
	float temp = 0.;
	
	/// compute total momentum
	for(i = 0; i < N; ++i) {
		Mi = M[type[i]];
		
		momentum.x += Mi * Vel[i].x;
		momentum.y += Mi * Vel[i].y;
		momentum.z += Mi * Vel[i].z;
		
		temp += Mi * (Vel[i].x*Vel[i].x + Vel[i].y*Vel[i].y + Vel[i].z*Vel[i].z); 
	}
	
	momentum.x /= N;
	momentum.y /= N;
	momentum.z /= N;
	
	temp /= kb * dim * N;
	
	/// remove each component of the total momentum to each particle
	for(i = 0; i < N; ++i) {
		Vel[i].x -= momentum.x/M[type[i]];
		Vel[i].y -= momentum.y/M[type[i]];
		Vel[i].z -= momentum.z/M[type[i]];
		
		Vel[i] *= sqrt(T0/temp);
	}
	
	/// overwrite velocities
	CudaSafeCall(cudaMemcpy( dev_Vel, Vel, N*sizeof(float4), cudaMemcpyHostToDevice ));
}


void util_rescaleVelocities(int N) {
	
	int nthreads = NTHREADS;                   /// Number of launch threads
	int nblocks = N / nthreads;                /// Only Synchronous blocks
	
	/// Compute the temperature of the system in two steps, first a block-wide reduction + thread-wide reduction
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_type, dev_Vel, N, dev_partialT);
	if(debug)
		CudaCheckError();
	gpu_totalTemp_kernel <<<1,64,64*sizeof(double)>>> (N, dev_partialT, dev_T);
	if(debug)
		CudaCheckError();
	
	gpu_thermostatBerendsen_kernel<<<nblocks,nthreads>>> (dev_Vel, dev_T, T0, dt, N);
	if(debug)
		CudaCheckError();
}


int util_countAdsorbed(void) 
{
	int i, typei;
	double r,rx,ry,rz;
	
	memset(nAds, 0, sizeof(nAds));
	memset(nHard, 0, sizeof(nHard));
	memset(nSoft1, 0, sizeof(nSoft1));
	memset(nSoft2, 0, sizeof(nSoft2));
	memset(nt, 0, sizeof(nt));
	
	float rHC, rSC1, rSC2;
	
	
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rHC = 1.5*R[typei];
		rSC1 = rHC + 2*R[typei];
		rSC2 = rSC1 + 2*R[typei];
		
		++ nt[typei];
		
		rx = Coord[i].x - hLx;
		ry = Coord[i].y - hLy;
		rz = Coord[i].z - hLz;
		
		r = sqrt(rx*rx + ry*ry + rz*rz) - R_NP;
		
		if(r < rHC) {         /// Proteins in the hard corona
			++ nHard[typei];
		}
		else if(r < rSC1) {    /// Proteins in the soft corona
			++ nSoft1[typei];
		}
		else if(r < rSC2) {    /// Proteins in the soft corona
			++ nSoft2[typei];
		}
	}
	
	
	float nAdsTot = 0.;
	
	for(typei = 0; typei < ntypes; ++typei) {
		nAds[typei] = nHard[typei] + nSoft1[typei] + nSoft2[typei];
		nAdsTot += nAds[typei];
	}
	
	return nAdsTot;
}

// TODO: THE LIMITS OF THE REACTION REGION SHOULD BE AN INPUT PARAMETER OF THE SIMULATION
void util_countInside(void) 
{
	int i, typei;
	double rx, ry, rz;
	
	/// initialize the counters
	memset(nInner, 0, sizeof(nInner));
	
	/// count how many proteins are inside the reaction region
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		if (rx > 0.1*L && rx < 0.9*L)
			if (ry > 0.1*L && ry < 0.9*L)
				if (rz > 0.1*L && rz < 0.9*L)
					nInner[typei] ++;
	}
	
	/// remove the adsorbed proteins from the count
	for(i = 0; i < ntypes; ++i)
		nInner[i] -= nAds[i];
}

// TODO: THE LIMITS OF THE REACTION REGION SHOULD BE AN INPUT PARAMETER OF THE SIMULATION
void util_setBuffer(void) {
	
	int i, typei;
	double rx, ry, rz;
	double VSys = pow(0.8*L, 3);
	int delta[ntypes];
	double Vout = pow(L, 3);
	
	for(typei = 0; typei < ntypes; ++typei) {
		
		/// Count the number of proteins in the BUFFER region
		
		nOuter[typei] = nt[typei] - nInner[typei] - nAds[typei];
		
		/// Count the equivalent BULK concentration of FREE proteins in REFERENCE system
		
		cTot[typei] = (nTot[typei] - nAds[typei]) / VTot;
		
		/// Count the equivalent BULK concentration in SIMULATION box
		
		cSys[typei] = nInner[typei] / VSys;
		
		delta[typei] = + (cTot[typei] - cSys[typei]);
		
		nFree[typei] = cTot[typei]*Vout;
		
		nBuff[typei] = (nt[typei] - nAds[typei] - nFree[typei] - delta[typei]*Vout);
	}
	
	/// count how many proteins of each type are of buffer type, and remove from the counting
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		/// check if it is in buffer zone
		if (rx < 0.1*L || rx > 0.9*L || ry < 0.1*L || ry > 0.9*L || rz < 0.1*L || rz > 0.9*L)
			if (isOutsideClosed[i])
				nBuff[typei] --;
	}
	
	double p[ntypes];
	
	for(typei = 0; typei < ntypes; ++typei)
		p[typei] = nBuff[typei] / nOuter[typei];
	
	/// count how many proteins are inside the reaction region
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		/// check if it is in buffer zone
		if (rx < 0.1*L || rx > 0.9*L || ry < 0.1*L || ry > 0.9*L || rz < 0.1*L || rz > 0.9*L) {
			if(isOutsideClosed[i] == 1 && nBuff[typei] < 0)
				if (rand()/(RAND_MAX+1.) < -p[typei]) {
					isOutsideClosed[i] = 0;
				}
			if(isOutsideClosed[i] == 0 && nBuff[typei] > 0)
				if (rand()/(RAND_MAX+1.) < p[typei]) {
					isOutsideClosed[i] = 1; /// WARNING THIS REMOVES THE BUFFER!!!
				}
		}
		/// if it is in the reaction zone
		else 
			isOutsideClosed[i] = 0;
	}
	
	CudaSafeCall(cudaMemcpy( dev_isOutsideClosed,  isOutsideClosed,  N*sizeof(int),   cudaMemcpyHostToDevice ));
}

void util_setBuffer_Equil(void) {
	
	int i, typei;
	double rx, ry, rz;
	double VSys = pow(0.8*L, 3);
	int delta[ntypes];
	double Vout = pow(L, 3);
	
	for(int typei = 0; typei < ntypes; ++typei) {
		/// Count the number of proteins in the BUFFER region
		nOuter[typei] = nt[typei] - nInner[typei] - nAds[typei];
		
		/// Count the equivalent BULK concentration of FREE proteins in REFERENCE system
		cTot[typei] = (nTot[typei] - nAds[typei]) / VTot;
		
		/// Count the equivalent BULK concentration in SIMULATION box 
		cSys[typei] = (nInner[typei]) / VSys;;
		
		delta[typei] = + (cTot[typei] - cSys[typei]);
		
		nFree[typei] = cTot[typei]*Vout;
		
		nBuff[typei] = (nt[typei] - nAds[typei] - nFree[typei] - delta[typei]*Vout);
	}
	
	/// count how many proteins of each type are of buffer type, and remove from the counting
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		/// check if it is in buffer zone
		if (rx < 0.1*L || rx > 0.9*L || ry < 0.1*L || ry > 0.9*L || rz < 0.1*L || rz > 0.9*L)
			if (isOutsideClosed[i])
				nBuff[typei] --;
	}
	
	double p[ntypes];
	
	for(int typei = 0; typei < ntypes; ++typei)
		p[typei] = nBuff[typei] / nOuter[typei];
	
	/// count how many proteins are inside the reaction region
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		/// check if it is in buffer zone
		if (rx < 0.1*L || rx > 0.9*L || ry < 0.1*L || ry > 0.9*L || rz < 0.1*L || rz > 0.9*L) {
			if(isOutsideClosed[i] == 1 && nBuff[typei] < 0)
				if (rand()/(RAND_MAX+1.) < -p[typei]) {
					isOutsideClosed[i] = 0;
				}
			if(isOutsideClosed[i] == 0 && nBuff[typei] > 0)
				if (rand()/(RAND_MAX+1.) < p[typei]) {
					isOutsideClosed[i] = 1; /// WARNING THIS REMOVES THE BUFFER
				}
		}
		/// if it is in the reaction zone
		else 
			isOutsideClosed[i] = 0;
	}
	
	for(int typei = 0; typei < ntypes; ++typei)
		printf("Conc. obj. type %d: %e\t Act. Conc.: %e\n", typei, cTot[typei], cSys[typei]);
	
	CudaSafeCall(cudaMemcpy( dev_isOutsideClosed,  isOutsideClosed,  N*sizeof(int),   cudaMemcpyHostToDevice ));
}


// TODO: THE LIMITS OF THE REACTION REGION SHOULD BE AN INPUT PARAMETER OF THE SIMULATION
void util_applyBufferConditions(void) {
	
	int i;
	double rx, ry, rz;
	
	for(i = 0; i < N; ++i) {
		rx = Coord[i].x;
		ry = Coord[i].y;
		rz = Coord[i].z;
		
		if (rx < 0.1*L || rx > 0.9*L || ry < 0.1*L || ry > 0.9*L || rz < 0.1*L || rz > 0.9*L) 
			isOutsideClosed[i] = 0;
		else
			isOutsideClosed[i] = 1;
	}
}

void util_fractionBound (double t) {
	
	/// in FCS the observation volume is ~ 1 um^3, so it is ~ (1000 nm)^3
	double Veff = pow(1000,3);
	/// volume of the reaction region in the system
	double VSys = pow(0.8*L, 3);
	double Conc[ntypes];
	double nEq[ntypes], fB[ntypes];
	
	/// compute the concentration of each protein in the reaction region
	for(int typei = 0; typei < ntypes; ++typei)
		Conc[typei] = nInner[typei]/VSys;
	
	/// compute the equivalent number of protein in an FCS observation cell of size Veff
	for(int typei = 0; typei < ntypes; ++typei)
		nEq[typei] = Conc[typei]*Veff;
	
	///the fraction bound is simply the ratio of adsorbed to (free+adsorbed) proteins
	for(int typei = 0; typei < ntypes; ++typei)
		fB[typei] = nAds[typei] /(nAds[typei] + nEq[typei]);
	
	FILE * myFile;
	char filename[100];
	
	snprintf(filename, sizeof(filename), "%s%s", root, "results/fractionBound/fractionBound.dat");
	
	myFile = fopen(filename, "a");
	
	fprintf(myFile, "%e\t", t);
	for(int typei = 0; typei < ntypes; ++typei)
		fprintf(myFile, "%e\t", fB[typei]);
	fprintf(myFile, "\n");
	fclose(myFile);
}


void util_printAdsorbed(double t, FILE * file) 
{
	fprintf(file,"%e\t", t);
	for(int typei = 0; typei < ntypes; ++typei) fprintf(file,"%d\t",nHard[typei]);
	for(int typei = 0; typei < ntypes; ++typei) fprintf(file,"%d\t",nSoft1[typei]);
	for(int typei = 0; typei < ntypes; ++typei) fprintf(file,"%d\t",nSoft2[typei]);
	fprintf(file,"\n");
	fflush(file);
	
	if(verbose) {
		printf("Total:         ");
		for(int typei = 0; typei < ntypes; ++typei) printf("%d\t", nt[typei]);
		printf("\n");
		
		printf("Hard Corona:   ");
		for(int typei = 0; typei < ntypes; ++typei) printf("%d\t", nHard[typei]);
		printf("\n");
		
		printf("Soft Corona 1: ");
		for(int typei = 0; typei < ntypes; ++typei) printf("%d\t", nSoft1[typei]);
		printf("\n");
		
		printf("Soft Corona 2: ");
		for(int typei = 0; typei < ntypes; ++typei) printf("%d\t", nSoft2[typei]);
		printf("\n");
	}
}


void util_printConcentration(double t, FILE * file) 
{
	fprintf(file, "%e\t", t);
	
	for(int typei = 0; typei < ntypes; ++typei)
		fprintf(file, "%e\t", cSys[typei]);
	for(int typei = 0; typei < ntypes; ++typei)
		fprintf(file, "%e\t", cTot[typei]);
	fprintf(file, "\n");
	fflush(file);
	
	if(verbose) 
		for(int typei = 0; typei < ntypes; ++typei)
			printf("Conc. obj. type %d: %e\t Act. Conc.: %e\n", typei, cTot[typei], cSys[typei]);
}


void util_addXYZframe (FILE * fp) 
{
	int i;
	
	/// The header of the frame is the total number of particles
	fprintf(fp,"%d\n\n",N+1);
	
	for(i = 0; i < N; ++i) 
		/// The format of each line is: Label X Y Z
		fprintf(fp,"atom%d\t%1.8f\t%1.8f\t%1.8f\n",type[i],Coord[i].x,Coord[i].y,Coord[i].z);
	
	/// the last atom is the NP
	fprintf(fp,"atom%d\t%1.8lf\t%1.8lf\t%1.8lf\n",3,x_NP,y_NP,z_NP);
	fflush(fp);
}


void init_rdf() 
{
	int i;
	
	rdfMin = R_NP;      /// Distancia zero de la g(r)
	rdfMax = (0.8*L)/2; /// Distancia maxima
	
	dr = 0.1;
	nrdf = (int) ((rdfMax - rdfMin)/dr) + 1;          /// Nombre de punts a cada g(r)
	
	for(i = 0; i < ntypes; ++i)
		rdf = (float *)calloc(sizeof(float),ntypes*nrdf);   /// Inicialitzacio de les g(r) a zero
}

void util_addToRdf (void) 
{
	float r,rx,ry,rz;
	int i;
	
	for(i = 0; i < N; ++i) 
	{
		rx = Coord[i].x - hLx;
		ry = Coord[i].y - hLy;
		rz = Coord[i].z - hLz;
		
		r = sqrt(rx*rx + ry*ry + rz*rz) - R_NP;
		
		if(r + R_NP < rdfMax) 
			rdf[type[i]*nrdf + (int)(r/dr + 0.5)] += 1;
	}
	
	++countRdf;
}

void util_calcPrintRdf (int ntimes) 
{
	int i, j;
	float r, V_NP, rho, rdfTot;
	
	V_NP = 4*M_PI/3 * R_NP*R_NP*R_NP;
	
	rho = N/(V - V_NP);
	
	FILE * file1;
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s", root, "results/rdf/rdf.dat");
	file1 = fopen(filename, "w");
	
	if (file1 == NULL) {
		printf("Error: file rdf.dat could not be opened.\n");
		exit(1);
	}
	
	for(i = 0; i < nrdf; ++i) 
	{
		r = i*dr + R_NP;
		
		fprintf(file1,"%e\t",i*dr);
		
		rdfTot = 0.;
		
		for(j = 0; j < ntypes; ++j) {
			rdf[j*nrdf + i] /= 4*M_PI * rho * r*r * dr; /// radial distrib. function definition
			rdf[j*nrdf + i] /= ntimes;                  /// time average
			rdfTot += rdf[j*nrdf + i];
		}
		
		fprintf(file1,"%e\t",rdfTot);
		for(j = 0; j < ntypes; ++j) 
			fprintf(file1,"%e\t",rdf[j*nrdf + i]);
		fprintf(file1,"\n");
	}
	
	fclose(file1);
}


