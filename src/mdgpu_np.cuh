#include <vector_types.h>
#include "cutil_math.h"
#include <curand_kernel.h>

#define EMPTY -1
#define TRUE   1
#define FALSE  0

/// This macro activates the check for errors with the GPU (may influence performance)
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

/// This macro activates the protein-protein interactions in the corona
//#define SC

/// HSA-SiO2
//#define DLVO_EDL_ENERGY 0.864f
/// Apo-SiO2
//#define DLVO_EDL_ENERGY 0.565f
/// Fib-SiO2
//#define DLVO_EDL_ENERGY 1.501f

#define ALPHA .4f
//#define BETA  10.0f


char root[100] = "./";


/// Definicio del nombre de passos en el loop principal
long int 	thermoSteps		= 10000L, 
		keepRatioPeriod	= 10000L, 
		rdfPeriod		= 1000L,
		moviePeriod		= 10000L;

long int 	equilTime		= 1000L,
		berendsenTime	= 1000L,
		nosehooverTime	= 1000L,
		relaxTime1		= 0L,
		relaxTime2		= 0L, 
		simulTime		= 500000001L, 
		rdfTime		= 50000000L;

/// Pas d'integracio, i control del temps de relaxament del termostat
float dt = 0.005, hdt = dt*0.5, * langPrefactor, Gamma1, Gamma2, Gamma3;

float Tinit = 1.0;     /// Temperatura inicial
float T0 = 1.0;     /// Temperatura objectiu
float Q0 = (0.5*0.5*3072*T0/(4*M_PI*M_PI));     /// parametre de coupling del termostat de Nose-Hoover. Ells proposen Q = 6NkT. Martyna et al proposen: Q = NkT tau^2 / 4 Pi^2, on tau es el periode de les oscilacions fonamentals del sistema ~ 10dt-100dt. Per exemple si dt=0.01 i tau=10dt, tau^2=0.01
float Q1 = (Q0/3072);     /// parametre de coupling del termostat de Nose-Hoover. Martyna et al proposen Qj = kT tau^2 / 4 Pi^2 (com Q0/N)
float Q2 = (Q1/4); /// The Mth thermostat oscillates with a frequency 2w. Like Q1/(2*2)
float tau = 10.;     /// parametre de relaxacio del termostat de Berendsen

int restart = 0;     /// Flag que indica si estem inciciant des d'una execucio previa
int gpuId = 0;       /// Identificador de la GPU a utilitzar
int verbose = 0;     /// Selector de mode 'verbose'
int constantConcentration = FALSE;

/// Nombre de threads a les rutines principals de la GPU. Per obtenir la
/// maxima eficiencia, el nombre de proteines ha de ser N = M * NTHREADS
/// i el nombre de threads multiple de 32.
/// Per exemple, si es volen simular 1000 particules, es millor simular-ne
/// N = 1024 = 4 * 256 = 4 * 8 * 32
int NTHREADS = 256;

/// Nombre maxim de veins a la llista de Verlet. Si es supera, el programa llança un avis
/// No hi ha problema en que aquest nombre sigui gran, simplement s'ocupara
/// mes memoria RAM/DRAM, ja es te en compte de passar el minim d'informacio en els cudaMemcpy
#define MAXNEIG 250

#define ntypes 3

/// Constant de Boltzmann i dimensio de l'espai
#define kb  1.
#define dim 3

/// Yoshida suzuki parameters with m=3, for the 3-chain Nose-Hoover thermostat
#define	w1	 1.35067597676
#define	w2	-1.70135195352
#define	w3	 1.35067597676


/// Comptador del nombre de cops que es recalcula la llista de verlet
int verletlistcount = 0;

/// Radi de la NP. Podria ser llegit del fitxer de config
float R_NP;
__device__ __constant__ float dev_R_NP;

/// Radi de cutoff i radi de l'esfera de veins de verlet. Aquests es redefineixen mes tard a initConfig
float rc, rv, rv2;
float rn, skin2, rc2, rc2_NP, R2;   /// Mes variables relacionades amb el tamany dels cutoff
float vec_rc2[ntypes], vec_rv2[ntypes];

/// Densitat, tamany lateral de la caixa i volum
double rho, box, hbox, Lx, Ly, Lz, hLx, hLy, hLz;
float4 Box;

int nRef0, nRef1, nRef2;

/// Temperatura instantania i Pressio
double T,P;
/// Energies
float Epot = 0, Etot;
double Ekin;
double chi0, chi1, chi2;
float velScale;
double * dev_partialT, * dev_T, * dev_Ekin, * dev_chi0, * dev_chi1, * dev_chi2/* * dev_scale*/;


/// Flag que controla en quins moments activar les funcions termodinamiques
int calc_thermo = 0;
long int updateVerletList;
int updateVerletListPeriod;

double logscale = 1.4;
long int computeLogScale;
long int offsetTime = 0;
int restartIndex = 0;

/// Nombre lateral de molecules inicial, i nombre de cel.les per la llista de veins
int nc, ncx, ncy, ncz, nct;

double eps = 1.0;
double epsNP = 100.;

double _4eps, _48eps;
double ecut = 0, fcut = 0;
double etail, ptail;

/// Debye-H"ukel screening length
double C_PBS = 0.001; 
float kappa;
double K_pp;
int epsDiag = 1;
float K;

/// Vector que conte el tipus de cada molecula
char * type;
char * dev_type;

int * isInsideClosed;
int * dev_isInsideClosed;
int * isOutsideClosed;
int * dev_isOutsideClosed;

long int longestStopTime = 0;
int * stopMol, * dev_stopMol;
long int * stopTimeMol;

int N;
float L, hL, V;

__device__ __constant__ float dev_L, dev_hL, dev_V;
__device__ __constant__ float4 dev_Box, dev_xyzNP;
__device__ __constant__ int dev_N;
__device__ __constant__ float dev_rc2_NP, dev_rv2, dev_skin2, dev_hdt, dev_dt, dev_langPrefactor[ntypes];
__device__ __constant__ float dev_vec_rc2[ntypes], dev_vec_rv2[ntypes];
__device__ __constant__ float dev_kappa;
__device__ __constant__ float dev_K;

float * R, * Rs;   /// Vectors dels radis de les molecules
float * M, * sqrM;         /// Vector de la massa de les molecules
float * EpsNP;     /// Vector de les energies d'interaccio proteina-NP
float * EpsEDL;   /// Vector de les energies d'interaccio del electrical double layer
float * EpsProt;   /// Vector de les energies d'interaccio proteina-proteina


__device__ __constant__ float dev_M[ntypes], dev_sqrM[ntypes], dev_R[ntypes], dev_Rs[ntypes], dev_EpsNP[ntypes], dev_EpsEDL[ntypes];

/// Vectors que acumulen els resultats de l'energia i la Pressio a cada block de la GPU, i que es sumen a la CPU
float * partial_Epot, * partial_P;
float * partial_Ekin, * partial_T;
float * dev_partial_Epot, * dev_partial_P, * dev_Epot, * dev_P;
float * Gamma;
float * dev_gamma;

/// Part d'Energia potencial degut a les interaccions molecula-NP 
float Epot_NP, P_NP;
float * dev_Epot_NP, * dev_P_NP;
float * dev_Epot_walls, * dev_P_walls;

/// Vectors de posicio
float4 * Coord;
float4 * dev_Coord;

int4 * Image, * dev_Image;

/// Vectors de les posicions de Verlet
float4 * CoordVerlet;
float4 * dev_CoordVerlet;

float x_NP, y_NP, z_NP;    /// Posicio de la NP
__device__ __constant__ float dev_x_NP,dev_y_NP,dev_z_NP;

float4 * Vel, * dev_Vel;   /// Vectors de velocitat
float4 * Acc, * dev_Acc;   /// Vectors d'acceleracions

float4 * RandomForce, * dev_Rnd;

/// Vectors de la linked list de cel.les pel calcul de la verlet list
int * head, * list;
int * dev_head, * dev_list;
/// Vectors de la llista de verlet
unsigned char * nlist;
int * vlist;
unsigned char * dev_nlist;
int * dev_vlist;
/// Nombre maxim de veins a la llista de verlet, es determina cada cop que es calcula la llista
unsigned char neigmax;
unsigned char * dev_neigmax;
/// Flag que avisa de quan cal recalcular la llista de verlet
unsigned int * dev_newlist;

int n0_0, n1_0, n2_0, n0_t, n1_t, n2_t, n0_ads, n1_ads, n2_ads, n0_hard, n1_hard, n2_hard, n0_soft1, n1_soft1, n2_soft1, n0_soft2, n1_soft2, n2_soft2;
int nt[ntypes], nHard[ntypes], nSoft1[ntypes], nSoft2[ntypes];

int nstopped;

int nAds[ntypes], nInner[ntypes], nRef[ntypes], nTot[ntypes];
double VTot;

int nOuter[ntypes];
double cTot[ntypes], cSys[ntypes];
double nBuff[ntypes], nFree[ntypes];

/// Parametred de la g(r) i el potencial tabulat
float dr, rdfMin, rdfMax;
float * rdf;
int nrdf;
int countRdf = 0;

/// tabulated potential and forces

int ntable;
__device__ __constant__ int dev_ntable;

float * tableU_rep,* tableU_att, * dev_tableU_rep,* dev_tableU_att;

float * tableF_rep,* tableF_att_symm, * tableF_att_noSymm, * dev_tableF_rep,* dev_tableF_att_symm, * dev_tableF_att_noSymm;

long int t;

///////////////// Function declarations ////////////////////////////////
/// Init
void initialize (int argc, char * argv[]);

/// Integrators
void verlet_integrateNVE (bool update);
void verlet_integrateBerendsenNVT (void);
void verlet_integrateNoseHooverNVT (bool equil, bool update);

void verlet_integrateLangevinNVT (bool computeMSD, bool update);
void verlet_integrateLangevinNVTequil (void);
void verlet_integrateLangevinNVTrelax (bool update);
void verlet_integrateLangevinNVTrelax_bak (bool update, int molid1, int molid2);

/// Thermodynamics
void thermo_compute(int npart);
void thermo_computeTemperature(void);

/// RDF
void init_rdf(void);
void util_addToRdf (void);
void util_calcPrintRdf (int ntimes);

/// Utils
void util_resetThermostat (void);
void util_calcVerletList (void);
void util_removeDrift(void);
void util_rescaleVelocities(int N);
void util_keepRatios(void);
void util_addXYZframe (FILE * fp);
 int util_countAdsorbed(void);
void util_countInside(void);
void util_setBuffer(void);
void util_setBuffer_Equil(void);
void util_applyBufferConditions(void);
void util_fractionBound (double t);
void util_printAdsorbed(double t, FILE * file);
void util_printConcentration(double t, FILE * file);
void util_saveState (long int t);
void util_cudaSetMeFree (void);
double util_gaussianRandom (double mu, double sig);

curandState* devStates;

int debug = 0;

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
#endif
	
	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
//	err = cudaDeviceSynchronize();
//	if( cudaSuccess != err )
//	{
//		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
//		file, line, cudaGetErrorString( err ) );
//		exit( -1 );
//	}
#endif
	 
	return;
}
