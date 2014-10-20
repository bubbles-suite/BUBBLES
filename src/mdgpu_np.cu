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

double util_gaussianRandom (double mu, double sig);

int main ( int argc, char * argv[] ) 
{
	initialize(argc, argv);        /// initialize simulation
	
	if(verbose)
		printf("Starting simulation with %d particles\n", N);
	
	/** Create the file that contains all the coordinates with all the frames of the movie */
	FILE * movie1, * movie2;
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s%d%s", root, "results/movie/movie_run", restartIndex, ".xyz");
	movie1 = fopen(filename,"w");
	snprintf(filename, sizeof(filename), "%s%s", root, "results/movie/movie.xyz");
	if(restart) movie2 = fopen(filename,"a");    /// open in append mode
	else        movie2 = fopen(filename,"w");    /// open in overwrite mode
	
	if (movie1 == NULL || movie2 == NULL) {
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
	
	printf("Stop mol 1: %d\n", (stopTimeMol[0] > 0 ? 1 : 0));
	printf("Stop mol 2: %d\n", (stopTimeMol[1] > 0 ? 1 : 0));
	printf("Stop mol 3: %d\n", (stopTimeMol[2] > 0 ? 1 : 0));
	
	printf("Stop mol time 1: %d\n", stopTimeMol[0]);
	printf("Stop mol time 2: %d\n", stopTimeMol[1]);
	printf("Stop mol time 3: %d\n", stopTimeMol[2]);
	
	////////////////// SIMULATION STARTS HERE //////////////////////////
	
	util_applyBufferConditions();
	util_countInside();
	util_setBuffer_Equil();
	
	if(!restart) {  /// First check if we are restarting from a previous run or not
		/// Add the first frame with the initial configuration to the movie file
		util_addXYZframe(movieLog);    
		util_addXYZframe(movie2); 
		
		/////////////////////// SET VERLET LIST UPDATE PERIOD //////////////
		
		/// After equilibration, run for a number of cycles to compute the Verlet List update period
		verletlistcount = 0;
		
		////////////////// FAST EQUILIBRATE INIT CONFIG ////////////////////
		
		/** Run the simulation for a number of steps of equilibration, without adsorption */
		for (t = 0; t < berendsenTime; ++t) {
			/// Integrator with Berendsen thermostat to set T0 rapidly
			verlet_integrateBerendsenNVT(); 
			
			if(t % thermoSteps == 0) {    /// Write to file periodically
				CudaSafeCall(cudaMemcpy( &T, dev_T, sizeof(double), cudaMemcpyDeviceToHost ));
				
				Ekin = 0.5*kb*T*dim*N;
				
				if(verbose)
					printf("Step %lu Config \t Etot %1.4f Epot %1.4f Ekin %1.4f P %1.4f T %1.4f\n",t,Etot/N,Epot/N,Ekin/N,P,T);
				
				fprintf(timeseries,"%lf %f\n",t*dt - berendsenTime*dt,T);
			}
			
			if(t % (keepRatioPeriod/10) == 0) {
				
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				if(verbose)
					printf("Setting concentrations\n");
				
				util_countAdsorbed();
				util_countInside();
				util_setBuffer_Equil();
			}
		}
		
		util_resetThermostat();
		
		fflush(timeseries);
		
		/// Period of Verlet List updates, no need to check maximum displacements anymore. Take a 60% for security
		updateVerletListPeriod = berendsenTime * .66 / verletlistcount;
		if(verbose)
			printf("Verlet list computed %d times. Update period is: %d\n",verletlistcount,updateVerletListPeriod);
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
				util_setBuffer();
				util_fractionBound(t*dt);
				
				util_printConcentration(t, concentration);
				
				/// Append to adsorption profile
				util_printAdsorbed(t*dt,adsorption);
				
				/// Save movie frame. Warning! it is very slow
				//util_addXYZframe(movie2);
			}
			
			/// Save movie frame. Warning! it is very slow and disk consuming
			if(t % moviePeriod == 0) {
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				util_addXYZframe(movie2);
			}
			
			///save logscale data
			if(t == computeLogScale) {
				cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
				
				/// Compute the number of adsorbed particles on-the-fly
				/// THIS COULD ALSO BE DONE INSIDE THE GPU!
				int adsorbed = util_countAdsorbed();
				
				util_countInside();
				util_setBuffer();
				util_fractionBound(t*dt);
				
				util_printConcentration(t, concentrationLog);
				
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
					
					CudaSafeCall(cudaMemcpy (Vel, dev_Vel, N*sizeof(float4), cudaMemcpyDeviceToHost));
					
					for(int i = 0; i < N; ++i) 
						if(type[i] == typei) {
							stopMol[i] = 0;
							
							double Mi = M[typei];
							
							Vel[i].x = util_gaussianRandom(0,1) * sqrt(kb*Tinit/Mi);
							Vel[i].y = util_gaussianRandom(0,1) * sqrt(kb*Tinit/Mi);
							Vel[i].z = util_gaussianRandom(0,1) * sqrt(kb*Tinit/Mi);
						}
					
					CudaSafeCall(cudaMemcpy( dev_stopMol,  stopMol,  N*sizeof(int),   cudaMemcpyHostToDevice));
					CudaSafeCall(cudaMemcpy (dev_Vel, Vel, N*sizeof(float4), cudaMemcpyHostToDevice));
					
					offsetTime = t;
					logscale = 1.4;
					computeLogScale = (long int)(offsetTime + logscale + 0.5);
					
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
			/// THIS COULD ALSO BE DONE INSIDE THE GPU!
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
			/// THIS COULD ALSO BE DONE INSIDE THE GPU!
			int adsorbed = util_countAdsorbed();
			
			util_countInside();
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
				
				float msd[3] = {0.,0.,0.};
				int num[3] = {0,0,0};
				
				for(int i=0; i < N; ++i) {
					float dx = fabs(Coord[i].x + Image[i].x*Lx - CoordIni[i].x);
					float dy = fabs(Coord[i].y + Image[i].y*Ly - CoordIni[i].y);
					float dz = fabs(Coord[i].z + Image[i].z*Lz - CoordIni[i].z);
					
					msd[type[i]] += dx*dx + dy*dy + dz*dz;
					
					num[type[i]] ++;
				}
				
				msd[0] /= num[0];
				msd[1] /= num[1];
				msd[2] /= num[2];
				
				fprintf(msdFile,"%lf\t%lf\t%lf\t%lf\n", (t-offsetTime+1)*dt, msd[0], msd[1], msd[2]);
				fflush(msdFile);
			}
			
			logscale *= 1.4;
			computeLogScale = (long int)(offsetTime + logscale + 0.5);
		}
		
		///save linear time data
		if(t % keepRatioPeriod == 0) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			
			/// Compute the number of adsorbed particles on-the-fly
			/// THIS COULD ALSO BE DONE INSIDE THE GPU!
			util_countAdsorbed();
			util_countInside();
			util_setBuffer();
			
			util_printConcentration(t, concentration);
		}
		
		/// Save movie frame. Warning! it is very slow and disk consuming
		if(t % moviePeriod == 0) {
			cudaMemcpy (Coord, dev_Coord, N*sizeof(float4), cudaMemcpyDeviceToHost);
			util_addXYZframe(movie2);
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
	fclose(movie1);
	fclose(movie2);
	fclose(adsorption);
	fclose(timeseries);
	
	/// Reset the GPU to clear profiling settings or allocated memory for example
	CudaSafeCall(cudaDeviceReset());
	
	return 0;
}


__global__ void gpu_GenerateVerletListSync_kernel(
	float4 * dPos, 
	int * vlist, 
	unsigned char * nlist, 
	int N) 
{
	extern __shared__ float4 sPos[];          /// allocate shared memory
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /** CONSTRAINT: threadIdMax must be a multiple of blockDim,
     * otherwise __synchthread() can not work! */
	float4 tPos = dPos[idx];               /// save reference particle position in register
	
	unsigned char neig = 0;                /// initialize the number of neighbors of reference particle
	
	/// loop over all the blocks, that can synchronize
	for(int j=0; j<gridDim.x; ++j) 
	{
		int bOffset = j * blockDim.x;       /// offset particle id of current block
		
		sPos[threadIdx.x] = dPos[bOffset + threadIdx.x];      /// fetch block of particle positions and save in shared memory
		
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
			
			/// Disable PERIODIC IMAGE IN Z DIRECTION if Wall
			if(d.z > dev_hL)
				d.z -= dev_L;
			else if(d.z < -dev_hL)
				d.z += dev_L;
			
			
			/// neighbor list condition
			if ( dev_vec_rv2[(int)tex1Dfetch(type_tex,idx)] > d.x*d.x + d.y*d.y + d.z*d.z ) {
				/// save neighbor id to verlet list and increase the number of neighbors
				vlist[neig*N + idx] = id;
				//vlist[idx*MAXNEIG + neig] = id;
				++ neig;
			}
		}
		
		__syncthreads();    /// Even if only one thread was succesful, all the other threads wait for it to finish
	}
	
	/// check if we missed the last block of particles
	if(gridDim.x*blockDim.x < N) {
		
		int tOffset = gridDim.x*blockDim.x;
		
		if(tOffset+threadIdx.x < N)
			sPos[threadIdx.x] = dPos[tOffset+threadIdx.x];      /// fetch block of particle positions and save in shared memory
		
		__syncthreads();
		
		/// loop over the particles in current block, until N
		for (int i = 0; i < N-tOffset; ++i) 
		{
			int id = tOffset + i;           /// current neighbor id = [tOffset,N)
			
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
			
			/// Disable PERIODIC IMAGE IN Z DIRECTION if Wall
			if(d.z > dev_hL)
				d.z -= dev_L;
			else if(d.z < -dev_hL)
				d.z += dev_L;
			
			
			/// neighbor list condition
			if ( dev_vec_rv2[(int)tex1Dfetch(type_tex,idx)] > d.x*d.x + d.y*d.y + d.z*d.z ) 
			{
				/// save neighbor id to verlet list and increase the number of neighbors
				vlist[neig*N + idx] = id;
				//vlist[idx*MAXNEIG + neig] = id;
				++ neig;
			}
		}
		
		__syncthreads();    /// Even if only one thread was succesful, all the other threads wait for it to finish
	}
	
	nlist[idx] = neig;        /// save the number of neighbors to global memory, colaesced
}

__global__ void gpu_GenerateVerletListAsync_kernel(
	float4 * dPos, 
	int * vlist, 
	unsigned char * nlist, 
	int N,
	int offset) 
{
	int idx = offset + threadIdx.x;
       
	float4 tPos = tex1Dfetch(Coord_tex,idx);   /// save reference particle position in register
	unsigned char neig = 0;
	
	for(int i = 0; i < N; ++ i)
	{
		float4 d = tPos - tex1Dfetch(Coord_tex,i);
		
		/// Image-corrected relative pair position 
		if(d.x > dev_hL)
			d.x -= dev_L;
		else if(d.x < - dev_hL)
			d.x += dev_L;
		
		if(d.y > dev_hL)
			d.y -= dev_L;
		else if(d.y < -dev_hL)
			d.y += dev_L;
		
		/// NO PERIODIC IMAGE IN Z DIRECTION if Wall
		if(d.z > dev_hL)
			d.z -= dev_L;
		else if(d.z < -dev_hL)
			d.z += dev_L;
		 
		if (dev_rv2 > d.x*d.x + d.y*d.y + d.z*d.z) {
			vlist[neig*N + idx] = i;
			//vlist[idx*MAXNEIG + neig] = i;
			++ neig;
		}
	 }
	
	__syncthreads();
	
	nlist[idx] = neig;          /// save the number of neighbors to global memory, colaesced
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
	gpu_GenerateVerletListSync_kernel <<<nblocks, nthreads, cacheMem>>> (dev_Coord,dev_vlist,dev_nlist,N); 
	if(debug)
		CudaCheckError();
	
	if(nblocks*nthreads < N)            /// Check for threads in Asynchronous blocks
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
	
	//printf("max %d\n", max);
	
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
	int 	i;
	double 	temp, v2;
	
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
	
	double coeff = sqrt(2*T0*Gamma/dt);
	int i;
	
	for(i = 0; i < N; ++i) {
		RandomForce[i] = coeff / sqrM[type[i]] * make_float4(util_uniformRandom(0,1),util_uniformRandom(0,1),util_uniformRandom(0,1),0);
	}
	
	CudaSafeCall(cudaMemcpy( dev_Rnd, RandomForce, N*sizeof(float4), cudaMemcpyHostToDevice ));
}




__global__ void gpu_RNG_setup ( curandState * state, unsigned long seed, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(id < N) {
	
	curand_init( (seed << 20) + id, 0, 0, &state[id]);
	
	id += blockDim.x*gridDim.x;
    }
} 

__global__ void gpu_RNG_generate ( curandState* globalState, float4 * Rnd, int N) 
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    while(ind < N) {
	    
	    int typei = (int)tex1Dfetch(type_tex,ind);
	    
	    __syncthreads();
	    
	    curandState localState = globalState[ind];
	    
	    /// uniform random in interval [-0.5:0.5] with zero mean and variance 1
	    
	    float4 randNum;
	    
	    randNum.x = curand_uniform( &localState ) - 0.5f;
	    randNum.y = curand_uniform( &localState ) - 0.5f;
	    randNum.z = curand_uniform( &localState ) - 0.5f;
	    randNum *= 3.4641f;
	    
	    randNum *= dev_langPrefactor / dev_sqrM[typei];
	    
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
		else if (!strcmp(str, "berendsenTime")){
			sscanf (line,"%s %ld", str, &berendsenTime);
			printf("number of berendsen thermostat timesteps set to: %ld\n", berendsenTime);
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
		else if (!strcmp(str, "Gamma")) {
			sscanf (line,"%s %f", str, &Gamma);
			printf("Langevin Heat-bath coupling Gamma=%f\n", Gamma);
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
	int nRef0, nRef1, nRef2;
	
	fscanf(file,"%d\t%lf\t%lf\t%f\t%f\t%d\t%d\t%d\n",&N,&C_PBS,&box,&V,&boxExp,&nRef0,&nRef1,&nRef2);    /// Read the detailed information of the box
	
	/// Debye-H"ukel screening length ([C_PBS] = M = 1 mol/l)
	kappa = 5.08 * sqrt(C_PBS);
	
	printf("Debye length: %lf\n", kappa);
	
	/// Reference Experimental values of the concentrations
	VTot = pow(boxExp, 3); /// Volume of a box containing exactly 1 NP
	
	/// Number of proteins inside a box of volume Vtot
	nTot[0] = nRef0; 
	nTot[1] = nRef1;
	nTot[2] = nRef2;
	
	N -= 1;                                      /// Remove the NP from the number of particles
	
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
	
	stopMol = (int *)calloc(N, sizeof(int));
	CudaSafeCall(cudaMalloc( (void**)&dev_stopMol, N*sizeof(int) ));
	
	/// Array with the mass of each type
	M = (float *)malloc(ntypes*sizeof(float));
	sqrM = (float *)malloc(ntypes*sizeof(float));
	
	/// Arrays with the radiuses of each type
	R = (float *)malloc(ntypes*sizeof(float));
	Rs = (float *)malloc(ntypes*sizeof(float));
	
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
	
	R[0] = R[1] = R[2] = 0.;
	Rs[0] = Rs[1] = Rs[2] = 0.;
	
	int * num = (int*)calloc(sizeof(int), ntypes);
	
	/// Read the init file
	for(i = 0; i < npart; ++i) 
	{
		fscanf(file,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",&typei,&xi,&yi,&zi,&Ri,&Rsi,&Mi,&epsi,&epsi2,&epsi3);
		
		type[i] = typei;
		
		num[typei] ++;
		
		/// Count how many particles of each type
		float r = sqrt((xi - hbox)*(xi - hbox) + (yi - hbox)*(yi - hbox) + (zi - hbox)*(zi - hbox));
		
		if(r > L/4.)
		{
			if(typei == 0)
				++ n0_0;
			else if(typei == 1)
				++ n1_0;
			else if(typei == 2)
				++ n2_0;
		}
		
		Coord[i] = make_float4(xi,yi,zi,0.f);
		
		R[typei] = Ri;
		Rs[typei] = Rsi;
		M[typei] = Mi;
		sqrM[typei] = sqrt(Mi);
		EpsNP[typei] = epsi;
		EpsEDL[typei] = epsi2;
		EpsProt[typei] = epsi3;
		
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
	
	for (i = 0; i < N; ++i) {
		int typei = type[i];
		
		if (stopTimeMol[typei] > 0) {
			stopMol[i] = 1;
			if (stopTimeMol[typei] > longestStopTime)
				longestStopTime = stopTimeMol[typei];
		}
	}
	
	/// Look for the maximum radius of the proteins
	float maxR = 0.;
	
	for(i = 0; i < ntypes; ++i)
		if(num[i] > 0 && maxR < Rs[i])
			maxR = Rs[i];
	
	/// Cutoff d'interaccio proteina-proteina
	vec_rc2[0] = (Rs[0] + maxR) + Rs[0];
	vec_rc2[1] = (Rs[1] + maxR) + Rs[1];
	vec_rc2[2] = (Rs[2] + maxR) + Rs[2];
	
	vec_rc2[0] *= vec_rc2[0];
	vec_rc2[1] *= vec_rc2[1];
	vec_rc2[2] *= vec_rc2[2];
	
	
	maxR = 0.;
	
	for(i = 0; i < ntypes; ++i)
		if(num[i] > 0 && maxR < vec_rc2[i])
			maxR = vec_rc2[i];
	
	rc = sqrt(maxR);
	
	if(verbose)
		printf("Parameter cutoff radius rc0: %1.3f, rc1: %1.3f, rc2: %1.3f\n", sqrt(vec_rc2[0]), sqrt(vec_rc2[1]), sqrt(vec_rc2[2]));
	
	rc2_NP = 5 * (2*rc);
	rc2_NP *= rc2_NP;
	
	/// Radi de l'esfera de verlet que conte els veins
	vec_rv2[0] = vec_rc2[0] * 1.5*1.5;
	vec_rv2[1] = vec_rc2[1] * 1.5*1.5;
	vec_rv2[2] = vec_rc2[2] * 1.5*1.5;
	
	if(verbose)
		printf("Parameter verlet radius rv0: %1.3f, rv1: %1.3f, rv2: %1.3f\n", sqrt(vec_rv2[0]), sqrt(vec_rv2[1]), sqrt(vec_rv2[2]));
	
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
	nlist = (unsigned char *)malloc(npart*sizeof(unsigned char));
	CudaSafeCall(cudaMalloc( (void **)&dev_nlist, npart*sizeof(unsigned char)));
	
	vlist = (int *)malloc(npart*MAXNEIG*sizeof(int));
	CudaSafeCall(cudaMalloc( (void **)&dev_vlist, npart*MAXNEIG*sizeof(int)));
	
	CudaSafeCall(cudaMalloc( (void **)&dev_neigmax, sizeof(unsigned char)));
	CudaSafeCall(cudaMalloc((void **)&dev_newlist,sizeof(unsigned int)));
	
	
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


__global__ void gpu_divideAccMass_kernel (float4 * Acc, int N) 
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	
	while(i < N)
	{
		Acc[i] /= dev_M[(int)tex1Dfetch(type_tex,i)];
		
		i += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_addRandomForces_kernel (float4 * Acc, float4 * Rnd, int N) 
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	
	while(i < N)
	{
		Acc[i] += Rnd[i];
		
		i += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_forceThreadPerAtomTabulatedEquilSync_kernel (
	float4 * Acc,
	unsigned char * nlist, 
	int * vlist,
	int N) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;     /// Cada thread s'encarrega del calcul de una particula
	
	unsigned char neighbors = nlist[i];        /// number of neighbors of current particle, coalesced read
	
	float4 ri = tex1Dfetch(Coord_tex,i);  /// eficient read, stored in texture cache for later scattered reads
	
	int typei = (int)tex1Dfetch(type_tex,i);
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		int j = vlist[neig*N + i];
		//int j = vlist[i*MAXNEIG + neig];
		
		float4 rj = tex1Dfetch(Coord_tex, j); /// Distance between 'i' and 'j' particles. RANDOM read
		float4 rij = rj - ri;
		
		rij.x -= dev_L * rintf(rij.x / dev_L);
		rij.y -= dev_L * rintf(rij.y / dev_L);
		rij.z -= dev_L * rintf(rij.z / dev_L);
		
		
		rij.w = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;       /// Save r2 = r*r in rij.w
		
		
		if ( rij.w < dev_vec_rc2[typei])   /// Skip current particle if not witihin cutoff
		{
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			/// Compute the acceleration in each direction
			/// Nomes per a les acceleracions de la particula del block actual
			/// Falta dividir per la massa
			acc += - tex1Dfetch(tableF_rep_tex,((int)(rij.w*100 + 0.5f) + dev_ntable*((int)tex1Dfetch(type_tex,j) + ntypes*typei))) * rij;
		}
	}
	
	Acc[i] = acc;        /// coalesced write to global memory
}




__global__ void gpu_forceThreadPerAtomTabulatedSync_kernel (
	float4 * Acc,
	unsigned char * nlist, 
	int * vlist,
	int N) 
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;     /// Cada thread s'encarrega del calcul de una particula
	
	unsigned char neighbors = nlist[i];        /// number of neighbors of current particle, coalesced read
	
	float4 ri = tex1Dfetch(Coord_tex,i);  /// eficient read, stored in texture cache for later scattered reads
	
	int typei = (int)tex1Dfetch(type_tex,i);
	
/// Activate with soft-corona interactions
#ifdef SC
	float4 di = ri - dev_xyzNP;
	di.w = sqrtf(di.x*di.x + di.y*di.y + di.z*di.z) - dev_R_NP;   /// Distance to the surface of the NP
#endif
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		int j = vlist[neig*N + i];
		//int j = vlist[i*MAXNEIG + neig];
		
		float4 rj = tex1Dfetch(Coord_tex, j); /// Distance between 'i' and 'j' particles. RANDOM read
		float4 rij = rj - ri;
		
		float pair_force_symm = 0.f;
#ifdef SC
		float pair_force_asymm = 0.f;
#endif
		
		
		rij.x -= dev_L * rintf(rij.x / dev_L);
		rij.y -= dev_L * rintf(rij.y / dev_L);
		rij.z -= dev_L * rintf(rij.z / dev_L);
		
		rij.w = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;       /// Save r2 = r*r in rij.w
		
		if ( rij.w < dev_vec_rc2[typei]) /// Skip current particle if not witihin cutoff
		{
			int typej = (int)tex1Dfetch(type_tex,j);
			
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			pair_force_symm = - tex1Dfetch(tableF_rep_tex,((int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)));
			
#ifdef SC
			float4 dj = rj - dev_xyzNP;
			dj.w = sqrtf(dj.x*dj.x + dj.y*dj.y + dj.z*dj.z) - dev_R_NP;
			
			/// Attractive interaction force, depends on the distances of i and j to te NP center
			/// Symmetric part
			//float expo = pow(dev_R[typei]*dev_R[typej]/(di.w*dj.w), ALPHA);
			float expo = exp(-di.w*dj.w/(K*K));
			
			pair_force_symm += expo * tex1Dfetch(tableF_att_symm_tex,((int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)));
			
			/// Non-symmetric part
			//pair_force_asymm = expo/di.w * tex1Dfetch(tableF_att_noSymm_tex,((int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)));
			
			pair_force_asymm = - expo*dj.w/(K*K) * tex1Dfetch(tableF_att_noSymm_tex,((int)(rij.w*100 + 0.5f) + dev_ntable*(typej + ntypes*typei)));
#endif
		
		/// Compute the acceleration in each direction
		/// Nomes per a les acceleracions de la particula del block actual
		/// Falta dividir per la massa
		acc += pair_force_symm * rij;
#ifdef SC
		acc += pair_force_asymm * di/(di.w+dev_R_NP); /// d/(d + Rnp) is a unitary vector in the direction of ri-Rnp
#endif
		}
	}
	
	Acc[i] = acc;        /// coalesced write to global memory
}


/// *** STILL UNDER CONSTRUCTION *** ///
__global__ void gpu_forceThreadPerAtomTabulatedAsync_kernel (
	float4 * Acc,
	unsigned char * nlist, 
	int * vlist,
	int N,
	int offset) 
{
	int i = offset + threadIdx.x;        /// Cada thread s'encarrega del calcul de una particula
	
	float pair_force;
	
	unsigned char neighbors = nlist[i];        /// number of neighbors of current particle, coalesced read
	
	int typei = (int)tex1Dfetch(type_tex,i);
	
	float4 ri = tex1Dfetch(Coord_tex,i);  /// eficient read, stored in texture cache for later scattered reads
	
	float4 di = ri - dev_xyzNP;
	di.w = sqrt(di.x*di.x + di.y*di.y + di.z*di.z) - dev_R_NP;   /// Distance to the center of the NP
	
	float4 acc = {0.0f, 0.0f, 0.0f, 0.f};
	
	/// Loop over the interacting neighbours
	for(unsigned char neig = 0; neig < neighbors; ++neig) 
	{
		/// Index of the neighbor. Coalesced read
		//int j = vlist[neig*N + i];
		
		float4 rj = tex1Dfetch(Coord_tex,vlist[neig*N + i]); /// Distance between 'i' and 'j' particles. RANDOM read
		//float4 rj = tex1Dfetch(Coord_tex,vlist[i*MAXNEIG + neig]); /// Distance between 'i' and 'j' particles. RANDOM read
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
		
		/// NO PERIODIC IMAGE IN Z DIRECTION if Wall
		if(rij.z > dev_hL)
			rij.z -= dev_L;
		else if(rij.z < -dev_hL)
			rij.z += dev_L;
		
		
		rij.w = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);       /// Save r2 = r*r in rij.w
		
		//if(rij.w < dev_rc2)            /// Skip current particle if not witihin cutoff
		if(rij.w < dev_vec_rc2[typei])
		{
			/** If the pair is interacting...
			 * Calculate the magnitude of the force DIVIDED BY THE DISTANCE */
			/// Repulsive interaction force
			pair_force = tex1Dfetch(tableF_rep_tex,(int)(rij.w*100 + 0.5f));
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
	
	Acc[i] = acc;        /// coalesced write to global memory
	
}

__device__ float gpu_calcPNPForce (float eps, float sigma2, float r2) 
{
	r2 = sigma2 / r2;      /// (sigma/r)**2
	
	float r12 = r2*r2;     /// (sigma/r)**4
	
	r12 = r12*r12*r12;     /// (sigma/r)**12
	
	return 4.f*24.f * eps * r2 * r12 * (r12 - 0.5) / sigma2;
}


__device__ float gpu_calcPNPForceEquil (float eps, float sigma2, float r2) 
{
	r2 = sigma2 / r2;      /// (sigma/r)**2
	
	float r12 = r2*r2;     /// (sigma/r)**4
	r12 = r12*r12*r12;     /// (sigma/r)**12
	
	return 4.f*24.f * eps * r2 * r12 * r12 / sigma2; /// Only repulsive, no adsorption possible
}

__global__ void gpu_forceNP_kernel (float4 * Acc, int N)
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
		
		int typei = (int)tex1Dfetch(type_tex,i);  /// Type of the current particle
		
		/// Properties of the given type
		EpsNPi = dev_EpsNP[typei];
		Ri = dev_R[typei];
		
		Coords = tex1Dfetch(Coord_tex,i);  /// Read partice positions. Coalesced read
		
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;  /// Distance to te NP
		 
		r = sqrt(r2);       /// Distance between the center of a protein and the NP
		r -= dev_R_NP;
		
		/// Rescale positions to shift the interaction potential
		Coords *= r/sqrt(r2);
		
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z; /// Squared distance of a protein to the NP's surface
		
		if(r2 < dev_rc2_NP)
		{
			pair_force = gpu_calcPNPForce(EpsNPi, Ri*Ri, r2);
			
			/// Update particle accelerations
			acc = pair_force * Coords;
		}
		
		if(i < threadIdMax)
			__syncthreads();
		
		Acc[i] += acc;               /// Coalesced write
		Acc[i] /= dev_M[typei];      /// Compute the resulting acceleration: a = F / m
		
		i += blockDim.x*gridDim.x;   /// Next particle
	}
}

__global__ void gpu_forceNP_DLVO_kernel (float4 * Acc, int N)
{
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/// Calculem la interaccio de cada particula amb la NP
	while (i < N) 
	{
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		int typei = (int)tex1Dfetch(type_tex,i);  /// Type of the current particle
		
		/// Properties of the given type
		float EpsNPi = dev_EpsNP[typei];
		float Ri = dev_R[typei];
		float EpsEDLi = dev_EpsEDL[typei];
		
		float4 Coords = tex1Dfetch(Coord_tex,i);  /// Read partice positions. Coalesced read
		
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		float r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;  /// Squared Distance to te NP
		
		float r1 = sqrt(r2);  /// Distance between the center of a protein and the NP
		
		float r = r1 - dev_R_NP;       /// Distance between the center of a protein and the surface of the NP
		
		float d = r - Ri;    /// Distance between the surface of a protein and the surface of the NP
		
		/// Rescale positions to shift the interaction potential
		Coords *= r/r1;
		
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z; /// Squared distance of a protein to the NP's surface
		
		
		/// DLVO VdW dispersion forces
		float pair_force = -EpsNPi/6.f * dev_R_NP*Ri / (dev_R_NP+Ri) / (d*d);
		
		/// DLVO Born repulsion
		pair_force += 7.f*6.f*EpsNPi * dev_R_NP*Ri / (dev_R_NP+Ri) * pow(0.5f, 6.f) / (7560.f * pow(d, 8.f));
		
		/// DLVO Electrical Double-layer forces
		pair_force += EpsEDLi * dev_kappa * exp(- dev_kappa * d);
		
		
		/// Update particle accelerations
		
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		Acc[i] += pair_force * Coords / sqrt(r2);               /// Coalesced write
		Acc[i] /= dev_M[typei];      /// Compute the resulting acceleration: a = F / m
		
		i += blockDim.x*gridDim.x;   /// Next particle
	}
}


__global__ void gpu_forceNP_Equil_kernel (float4 * Acc, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/// Calculem la interaccio de cada particula amb la NP
	while (i < N) 
	{
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		int typei = (int)tex1Dfetch(type_tex,i);  /// Type of the current particle
		
		/// Properties of the given type
		float EpsNPi = dev_EpsNP[typei];
		float Ri = dev_R[typei];
		
		float4 Coords = tex1Dfetch(Coord_tex,i);  /// Read partice positions. Coalesced read
		
		/// Coordinates relative to the center of the NP
		Coords.x -= dev_x_NP;
		Coords.y -= dev_y_NP;
		Coords.z -= dev_z_NP;
		
		float r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z;  /// Squared distance to te NP
		 
		float r = sqrtf(r2);       /// Distance between the center of a protein and the center of the NP
		r -= dev_R_NP;       /// Distance between the center of a protein and the surface of the NP
		
		/// Rescale the vector distance to shift the interaction potential
		Coords *= r/sqrtf(r2);
		
		r2 = Coords.x*Coords.x + Coords.y*Coords.y + Coords.z*Coords.z; /// Squared distance of a protein to the NP's surface
		
		float pair_force = gpu_calcPNPForceEquil(EpsNPi, Ri*Ri, r2);
		
		if(i < N/blockDim.x * blockDim.x)
			__syncthreads();
		
		Acc[i] += pair_force * Coords;               /// Coalesced write
		Acc[i] /= dev_M[typei];      /// Compute the resulting acceleration: a = F / m
		
		i += blockDim.x*gridDim.x;   /// Next particle
	}
}



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
		gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	else {
		gpu_forceThreadPerAtomTabulatedSync_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N, nblocks*nthreads);
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
		gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
	else
		gpu_forceNP_DLVO_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
	
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
	gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N);
	if(debug)
		CudaCheckError();
	
	if(nblocks*nthreads < N) {      /// Asynchronous block
		gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N, nblocks*nthreads);
		if(debug)
		CudaCheckError();
	}
	
	if(calc_thermo) {
		CudaSafeCall(cudaMemset(dev_Epot_NP, 0, sizeof(float)));
		CudaSafeCall(cudaMemset(dev_P_NP,    0, sizeof(float)));
	}
	
	/// Add to the forces the contribution of the NP in a separate calculation
	gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
	
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
		gpu_forceThreadPerAtomTabulatedEquilSync_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	else {
		gpu_forceThreadPerAtomTabulatedSync_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N);
		if(debug)
		CudaCheckError();
		
		if(nblocks*nthreads < N) {      /// Asynchronous block
			gpu_forceThreadPerAtomTabulatedAsync_kernel <<<1,N-nblocks*nthreads>>> (dev_Acc, dev_nlist, dev_vlist, N, nblocks*nthreads);
			if(debug)
		CudaCheckError();
		}
	}
	
	//if(calc_thermo) {
	//	CudaSafeCall(cudaMemset(dev_Epot_NP, 0, sizeof(float)));
	//	CudaSafeCall(cudaMemset(dev_P_NP,    0, sizeof(float)));
	//}
	//force_wall1_kernel <<<(N+191)/192,192>>> (/*dev_z,*/ dev_R, dev_M, dev_EpsNP, dev_az, N, Lz, epsNP, T0);	
	//force_wall2_kernel <<<(N+191)/192,192>>> (/*dev_z,*/ dev_R, dev_M, dev_EpsNP, dev_az, N, Lz, epsNP, T0);
	
	/// Add to the forces the contribution of the NP in a separate calculation
	if(equil)
		gpu_forceNP_Equil_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
	else 
		gpu_forceNP_DLVO_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
	//CudaCheckError();
	
	//util_generateRandomForces();
	//CudaCheckError();
	
	gpu_RNG_generate <<<nblocks,nthreads>>> ( devStates, dev_Rnd, N); 
	
	gpu_addRandomForces_kernel <<<nblocks,nthreads>>> (dev_Acc, dev_Rnd, N);
	if(debug)
		CudaCheckError();
	
//	gpu_divideAccMass_kernel <<<nblocks,nthreads>>> (dev_Acc, N);
//	CudaCheckError();
	
	/// Divide force over mass to obtain accelerations
	/// Now, It is done at the end of force_NP_kernel! 
	//divide_acc_mass_kernel <<<(N+191)/192,192>>> (/*dev_ax, dev_ay, dev_az,*/ dev_Acc/*,  dev_type, N*/);
}

void util_calcPPForceTable (void) 
{
	//ntable = (int)(rc * 10000 + 1);
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
		float d   = 1.1*(R[t1] + R[t2]);
		float D   = Rs[t1] + Rs[t2];
		
		
		
		for(double r2 = 0; r2 < rc*rc; r2+=1./100)
		{
			int index = (int)(r2*100+0.5) + ntable*(t2 + ntypes*t1);
			
			double r = sqrt(r2);
			
			if(r > d*0.5)
			{
				/// Correct barrier for correct diffusion: 2.45
				tableU_rep[index] = pow(d/r,24) + 2. / (1 + exp(30 * (r - D)/d));
				
				tableF_rep[index] = 24 * pow(d/r,24) / (r*r) + 2. * 30. / (r * d * pow(2*cosh(0.5 * 30 * (r - D)/d), 2));
				
				//tableF_att_symm[index] = eps * (r - 1.1*d) / (w*w) * exp(-pow(r - 1.1*d,2)/(2*w*w)) / r;
				tableF_att_symm[index] = -eps * (r - d)/(w*w) * exp(-pow(r - d,2)/(2*w*w)) / r;
				
				tableF_att_noSymm[index] = -eps * exp(-pow(r - d,2)/(2*w*w));
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
	
	if(restart)                           /// Check if we are starting from a previous run
		util_loadConfig();
	
	if(verbose) {
		printf("Configuration read\n");
		printf("Number of particles: %d\n", N);
	}
	
//	util_calcPPpotentialTable();                 /// Compute the potential tables
	util_calcPPForceTable();                 /// Compute the force tables
	
	T = thermo_temperature();         /// Compute the temperature
	if(verbose)
		printf("Temperature: %lf\n\n",T);
	
	
	langPrefactor = sqrt(2.*T0*Gamma/dt);
	
	////////////// Copy system state from CPU to GPU ///////////////////
	/// Dynamic variables
	CudaSafeCall(cudaMemcpy( dev_Coord, Coord, N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_Vel,   Vel,   N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_Acc,   Acc,   N*sizeof(float4), cudaMemcpyHostToDevice ));
	CudaSafeCall(cudaMemcpy( dev_type,  type,  N*sizeof(char),   cudaMemcpyHostToDevice ));
	
	CudaSafeCall(cudaMemcpy( dev_stopMol,  stopMol,  N*sizeof(int),   cudaMemcpyHostToDevice ));
	
	/// Assign textures
	CudaSafeCall(cudaBindTexture(NULL, Coord_tex, dev_Coord, N*sizeof(float4)));
	CudaSafeCall(cudaBindTexture(NULL, type_tex,  dev_type,  N*sizeof(char)));
	
	/// Set constant parameters in GPU
	CudaSafeCall(cudaMemcpyToSymbol(dev_N, &N, sizeof(int)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_L,  (&L),  sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_hL, (&hL), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_V, (&V), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_kappa, (&kappa), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_rc2_NP, (&rc2_NP), sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_rv2,    (&rv2),    sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_skin2,  (&skin2),  sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_vec_rc2,  vec_rc2,    ntypes*sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_vec_rv2,  vec_rv2,    ntypes*sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_dt,  (&dt),  sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_hdt, (&hdt), sizeof(float)));
	
	CudaSafeCall(cudaMemcpyToSymbol(dev_gamma, (&Gamma), sizeof(float)));
	CudaSafeCall(cudaMemcpyToSymbol(dev_langPrefactor, (&langPrefactor), sizeof(float)));
	
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
	
//	CudaSafeCall(cudaBindTexture(NULL, tableU_rep_tex, dev_tableU_rep, ntable*sizeof(float)));
//	CudaSafeCall(cudaBindTexture(NULL, tableU_att_tex, dev_tableU_att, ntable*sizeof(float)));
	
	/// Force tables
	CudaSafeCall(cudaMemcpy(dev_tableF_rep,        tableF_rep,        ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_tableF_att_symm,   tableF_att_symm,   ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dev_tableF_att_noSymm, tableF_att_noSymm, ntypes*ntypes*ntable*sizeof(float), cudaMemcpyHostToDevice));
	
	CudaSafeCall(cudaBindTexture(NULL, tableF_rep_tex,        dev_tableF_rep,        ntypes*ntypes*ntable*sizeof(float)));
	CudaSafeCall(cudaBindTexture(NULL, tableF_att_symm_tex,   dev_tableF_att_symm,   ntypes*ntypes*ntable*sizeof(float)));
	CudaSafeCall(cudaBindTexture(NULL, tableF_att_noSymm_tex, dev_tableF_att_noSymm, ntypes*ntypes*ntable*sizeof(float)));
	
	////////////////////////// Initialization //////////////////////////
	init_rdf() ;
	
	/// Compute the Verlet List at the beginning 
	util_calcVerletList();
	
	/// Then compute the forces, before first Coord and Vel update
	verlet_calcForce(TRUE);
}

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
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] += Acc[tid] * dev_hdt;
		Vel[tid] /= 1.f + dev_gamma * dev_hdt;
		
		tid += blockDim.x*gridDim.x;
	}
}

__global__ void gpu_updateVelocitiesLangevinRelax_kernel (
	float4 * Vel, 
	float4 * Acc, 
	int * stopMol,
	int N) 
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < N) 
	{
		Vel[tid] += Acc[tid] * dev_hdt;
		Vel[tid] /= 1.f + dev_gamma * dev_hdt;
		
		if(stopMol[tid])
			Vel[tid] = make_float4(0.f,0.f,0.f,0.f);
		
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
	
	float dev_LbuffMin = dev_L*0.25;
	float dev_LbuffMax = dev_L*0.75;
	
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
		
		float4 vel = (1.f - dev_gamma * dev_hdt) * Vel[tid];                     /// Coalesced reads
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

__global__ void gpu_updateVelocitiesPositionsLangevinMSD_kernel (
	float4 * Vel,
	float4 * Acc,
	float4 * CoordVerlet,
	float4 * Coord,
	int4 * image,
	int * isOutsideClosed,
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
		
		float4 vel = (1.f - dev_gamma * dev_hdt) * Vel[tid];                     /// Coalesced reads
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
	int * stopMol,
	int * isOutsideClosed,
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
		
		vel = (1.f - dev_gamma * dev_hdt) * Vel[tid];                     /// Coalesced reads
		vel += Acc[tid] * dev_hdt;          /// Advance velocity
		
		if(stopMol[tid])
			vel = make_float4(0.f,0.f,0.f,0.f);
		
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
		vel.w *= dev_M[tex1Dfetch(type_tex,tid)];              /// Multiply by the mass
		
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
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_Vel, N, dev_partialT);
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
		gpu_updateVelocitiesPositionsLangevinMSD_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_Image, dev_isOutsideClosed, N);
	else
		gpu_updateVelocitiesPositionsLangevin_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_isOutsideClosed, N);
	
	if(debug)
		CudaCheckError();
	
	if(updateVerletList)
		util_calcVerletList();
	
	verlet_calcForceLangevin (FALSE);     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocitiesLangevin_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, N);
	
	if(debug)
		CudaCheckError();
}

void verlet_integrateLangevinNVTrelax (bool updateVerletList) 
{
	/// IMPORTANT DE COMPROVAR EL NOMBRE DE THREADS I BLOCKS, I FER TESTS!!!
	int nthreads = NTHREADS;
	int nblocks = (N + nthreads - 1) / nthreads;
	
	/// First half-step
	gpu_updateVelocitiesPositionsLangevinRelax_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_CoordVerlet, dev_Coord, dev_stopMol, dev_isOutsideClosed, N);
	
	if(debug)
		CudaCheckError();
	
	if(updateVerletList)
		util_calcVerletList();
	
	verlet_calcForceLangevin (FALSE);     /// The acceleration a(t + dt) due to new positions and Force field
	
	/// Second half-step
	gpu_updateVelocitiesLangevinRelax_kernel <<<nblocks,nthreads>>> (dev_Vel, dev_Acc, dev_stopMol, N);
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
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_Vel, N, dev_partialT);
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
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_Vel,N,dev_partialT);
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
	gpu_partialTemp_kernel <<<64,128,128*sizeof(double)>>> (dev_Vel, N, dev_partialT);
	if(debug)
		CudaCheckError();
	gpu_totalTemp_kernel <<<1,64,64*sizeof(double)>>> (N, dev_partialT, dev_T);
	if(debug)
		CudaCheckError();
	
	gpu_thermostatBerendsen_kernel<<<nblocks,nthreads>>> (dev_Vel, dev_T, T0, dt, N);
	if(debug)
		CudaCheckError();
}


// TODO: MODIFY THIS FUNCTION TO BE SCALABLE TO NTYPES OF PROTEINS
int util_countAdsorbed(void) 
{
	int i, typei;
	double r,rx,ry,rz;
	
	memset(nAds, 0, sizeof(nAds));
	
	n0_hard = n1_hard = n2_hard = 0;
	n0_soft1 = n1_soft1 = n2_soft1 = 0;
	n0_soft2 = n1_soft2 = n2_soft2 = 0;
	n0_t    = n1_t    = n2_t    = 0;
	
	float rHC, rSC1, rSC2;
	
	for(i = 0; i < N; ++i)
	{
		typei = type[i];
		
		rHC = 1.5*R[typei];
		rSC1 = rHC + 2*R[typei];
		rSC2 = rSC1 + 2*R[typei];
		
		if(typei == 0)
			++ n0_t;
		else if(typei == 1)
			++ n1_t;
		else if(typei == 2)
			++ n2_t;
		
		rx = Coord[i].x - hLx;
		ry = Coord[i].y - hLy;
		rz = Coord[i].z - hLz;
		
		r = sqrt(rx*rx + ry*ry + rz*rz) - R_NP;
		
		if(r < rHC)          /// Proteins in the hard corona
		{
			if(typei == 0)
				++ n0_hard;
			else if(typei == 1)
				++ n1_hard;
			else if(typei == 2)
				++ n2_hard;
		}
		
		else if(r < rSC1) {    /// Proteins in the soft corona
			if(typei == 0)
				++ n0_soft1;
			else if(typei == 1)
				++ n1_soft1;
			else if(typei == 2)
				++ n2_soft1;
		}
		else if(r < rSC2) {    /// Proteins in the soft corona
			if(typei == 0)
				++ n0_soft2;
			else if(typei == 1)
				++ n1_soft2;
			else if(typei == 2)
				++ n2_soft2;
		}
	}
	
	nAds[0] = n0_hard + n0_soft1 + n0_soft2;
	nAds[1] = n1_hard + n1_soft1 + n1_soft2;
	nAds[2] = n2_hard + n2_soft1 + n2_soft2;
	
	return nAds[0] + nAds[1] + nAds[2];
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
// TODO: MODIFY THIS FUNCTION TO BE SCALABLE TO NTYPES OF PROTEINS
void util_setBuffer(void) {
	
	int i, typei;
	double rx, ry, rz;
	
	/// Count the number of proteins in the BUFFER region
	nOuter[0] = n0_t - nInner[0] - nAds[0];
	nOuter[1] = n1_t - nInner[1] - nAds[1];
	nOuter[2] = n2_t - nInner[2] - nAds[2];
	
	/// Count the equivalent BULK concentration of FREE proteins in REFERENCE system
	cTot[0] = (nTot[0] - nAds[0]) / VTot;
	cTot[1] = (nTot[1] - nAds[1]) / VTot;
	cTot[2] = (nTot[2] - nAds[2]) / VTot;
	
	double VSys = pow(0.8*L, 3);
	
	/// Count the equivalent BULK concentration in SIMULATION box 
	cSys[0] = nInner[0] / VSys;
	cSys[1] = nInner[1] / VSys;
	cSys[2] = nInner[2] / VSys;
	
	int delta[3];
	
	delta[0] = + (cTot[0] - cSys[0]);
	delta[1] = + (cTot[1] - cSys[1]);
	delta[2] = + (cTot[2] - cSys[2]);
	
	double Vout = pow(L, 3);
	
	nFree[0] = cTot[0]*Vout;
	nFree[1] = cTot[1]*Vout;
	nFree[2] = cTot[2]*Vout;
	
	nBuff[0] = (n0_t - nAds[0] - nFree[0] - delta[0]*Vout);
	nBuff[1] = (n1_t - nAds[1] - nFree[1] - delta[1]*Vout);
	nBuff[2] = (n2_t - nAds[2] - nFree[2] - delta[2]*Vout);
	
	
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
	
	double p[3];
	
	p[0] = nBuff[0] / nOuter[0];
	p[1] = nBuff[1] / nOuter[1];
	p[2] = nBuff[2] / nOuter[2];
	
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
					isOutsideClosed[i] = 1;
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
	
	/// Count the number of proteins in the BUFFER region
	nOuter[0] = n0_t - nInner[0] - nAds[0];
	nOuter[1] = n1_t - nInner[1] - nAds[1];
	nOuter[2] = n2_t - nInner[2] - nAds[2];
	
	/// Count the equivalent BULK concentration of FREE proteins in REFERENCE system
	cTot[0] = (nTot[0] - nAds[0]) / VTot;
	cTot[1] = (nTot[1] - nAds[1]) / VTot;
	cTot[2] = (nTot[2] - nAds[2]) / VTot;
	
	double VSys = pow(0.8*L, 3);
	
	/// Count the equivalent BULK concentration in SIMULATION box 
	cSys[0] = (nInner[0]) / VSys;
	cSys[1] = (nInner[1]) / VSys;
	cSys[2] = (nInner[2]) / VSys;
	
	int delta[3];
	
	delta[0] = + (cTot[0] - cSys[0]);
	delta[1] = + (cTot[1] - cSys[1]);
	delta[2] = + (cTot[2] - cSys[2]);
	
	double Vout = pow(L, 3);
	
	nFree[0] = cTot[0]*Vout;
	nFree[1] = cTot[1]*Vout;
	nFree[2] = cTot[2]*Vout;
	
	nBuff[0] = (n0_t - nAds[0] - nFree[0] - delta[0]*Vout);
	nBuff[1] = (n1_t - nAds[1] - nFree[1] - delta[1]*Vout);
	nBuff[2] = (n2_t - nAds[2] - nFree[2] - delta[2]*Vout);
	
	
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
	
	double p[3];
	
	p[0] = nBuff[0] / nOuter[0];
	p[1] = nBuff[1] / nOuter[1];
	p[2] = nBuff[2] / nOuter[2];
	
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
					isOutsideClosed[i] = 1;
				}
		}
		/// if it is in the reaction zone
		else 
			isOutsideClosed[i] = 0;
		
		if(stopMol[i])
			isOutsideClosed[i] = 1;
	}
	
	printf("concentracio objectiu 0: %e\t concentracio actual: %e\n", cTot[0], cSys[0]);
	printf("concentracio objectiu 1: %e\t concentracio actual: %e\n", cTot[1], cSys[1]);
	printf("concentracio objectiu 2: %e\t concentracio actual: %e\n", cTot[2], cSys[2]);
	//printf("L %lf nInner %d nAds %d nOuter %d nBuff %lf\n", L, nInner[2], nAds[2], nOuter[2], nBuff[2]);
	
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
			isOutsideClosed[i] = 1;
		else
			isOutsideClosed[i] = 0;
	}
}

// TODO: MODIFY THIS FUNCTION TO BE SCALABLE TO NTYPES OF PROTEINS
void util_fractionBound (double t) {
	
	/// in FCS the observation volume is ~ 1 um^3, so it is ~ (1000 nm)^3
	double Veff = pow(1000,3);
	/// volume of the reaction region in the system
	double VSys = pow(0.9*L, 3);
	double Conc[3];
	double nEq[3], fB[3];
	
	/// compute the concentration of each protein in the reaction region
	Conc[0] = nInner[0]/VSys;
	Conc[1] = nInner[1]/VSys;
	Conc[2] = nInner[2]/VSys;
	
	/// compute the equivalent number of protein in an FCS observation cell of size Veff
	nEq[0] = Conc[0]*Veff;
	nEq[1] = Conc[1]*Veff;
	nEq[2] = Conc[2]*Veff;
	
	///the fraction bound is simply the ratio of adsorbed to (free+adsorbed) proteins
	
	fB[0] = nAds[0] /(nAds[0] + nEq[0]);
	fB[1] = nAds[1] /(nAds[1] + nEq[1]);
	fB[2] = nAds[2] /(nAds[2] + nEq[2]);
	
	FILE * myFile;
	char filename[100];
	
	snprintf(filename, sizeof(filename), "%s%s", root, "results/fractionBound/fractionBound.dat");
	
	myFile = fopen(filename, "a");
	
	fprintf(myFile, "%e\t%e\t%e\t%e\n", t, fB[0], fB[1], fB[2]);
	
	fclose(myFile);
}

// TODO: MODIFY THIS FUNCTION TO BE SCALABLE TO NTYPES OF PROTEINS
void util_printAdsorbed(double t, FILE * file) 
{
	fprintf(file,"%e\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",t,n0_hard,n1_hard,n2_hard,n0_soft1,n1_soft1,n2_soft1,n0_soft2,n1_soft2,n2_soft2);
	fflush(file);
	
	if(verbose) {
		printf("Total:         %d\t%d\t%d\n", n0_t,    n1_t,    n2_t   );
		printf("Hard Corona:   %d\t%d\t%d\n", n0_hard,  n1_hard,  n2_hard );
		printf("Soft Corona 1: %d\t%d\t%d\n", n0_soft1, n1_soft1, n2_soft1);
		printf("Soft Corona 2: %d\t%d\t%d\n", n0_soft2, n1_soft2, n2_soft2);
	}
}


// TODO: MODIFY THIS FUNCTION TO BE SCALABLE TO NTYPES OF PROTEINS
void util_printConcentration(double t, FILE * file) 
{
	fprintf(file, "%e\t%e\t%e\t%e\t%e\t%e\t%e\n", t, cSys[0], cSys[1], cSys[2], cTot[0], cTot[1], cTot[2]);
	fflush(file);
	
	if(verbose) {
		printf("concentracio objectiu 0: %e\t concentracio actual: %e\n", cTot[0], cSys[0]);
		printf("concentracio objectiu 1: %e\t concentracio actual: %e\n", cTot[1], cSys[1]);
		printf("concentracio objectiu 2: %e\t concentracio actual: %e\n", cTot[2], cSys[2]);
	}
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
	rdfMax = L/2; /// Distancia maxima
	
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
	float r, V_NP, rho;
	
	V_NP = 4*M_PI/3 * R_NP*R_NP*R_NP;
	
	rho = N/(V - V_NP);
	
	FILE * file1, * file2;
	char filename[100];
	snprintf(filename, sizeof(filename), "%s%s%d%s", root, "results/rdf/rdf_run", restartIndex, ".dat");
	file1 = fopen(filename, "w");
	snprintf(filename, sizeof(filename), "%s%s", root, "results/rdf/rdf.dat");
	file2 = fopen(filename, "w");
	
	if (file1 == NULL || file2 == NULL) {
		printf("Error: file rdf.dat could not be opened.\n");
		exit(1);
	}
	
	for(i = 0; i < nrdf; ++i) 
	{
		r = i*dr + R_NP;
		
		for(j = 0; j < ntypes; ++j) {
			rdf[j*nrdf + i] /= 4*M_PI * rho * r*r * dr; /// radial distrib. function definition
			rdf[j*nrdf + i] /= ntimes;                  /// time average
		}
		
		fprintf(file1,"%e\t%e\t%e\t%e\t%e\n",i*dr,rdf[0*nrdf + i] + rdf[1*nrdf + i] + rdf[2*nrdf + i],rdf[0*nrdf + i],rdf[1*nrdf + i],rdf[2*nrdf + i]);
		fprintf(file2,"%e\t%e\t%e\t%e\t%e\n",i*dr,rdf[0*nrdf + i] + rdf[1*nrdf + i] + rdf[2*nrdf + i],rdf[0*nrdf + i],rdf[1*nrdf + i],rdf[2*nrdf + i]);
	}
	
	fclose(file1);
	fclose(file2);
}


