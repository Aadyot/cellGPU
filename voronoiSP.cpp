// Standard project-wide include
#include "std_include.h"
#include <random>

// CUDA headers
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

// Project-specific headers
#include "Simulation.h"
#include "voronoiQuadraticEnergy.h"
#include "selfPropelledParticleDynamics.h"
#include "simpleVoronoiDatabase.h"
#include "logEquilibrationStateWriter.h"
#include "analysisPackage.h"

/*!
This file sets up a simulation of a 2D Voronoi model with self-propelled particles,
using log-spaced time sampling for saving simulation states. It is based on voronoi.cpp,
but uses selfPropelledParticleDynamics instead of a Nose-Hoover thermostat.

NOTE that in the output, the forces and the positions are not, by default, synchronized! The NcFile
records the force from the last time "computeForces()" was called, and generally the equations of motion will 
move the positions. If you want the forces and the positions to be sync'ed, you should call the
Voronoi model's computeForces() funciton right before saving a state.
*/
int main(int argc, char*argv[])
{
    //...some default parameters
    int numpts = 200; //number of cells
    int USE_GPU = 0; //0 or greater uses a gpu, any negative number runs on the cpu
    int c;
    int tSteps = 5; //number of time steps to run after initialization
    int initSteps = 1; //number of initialization steps
    int reproducible_flag = 1; // 1 for reproducible, 0 for random

    double dt = 0.01; //the time step size
    double p0 = 3.8;  //the preferred perimeter
    double a0 = 1.0;  // the preferred area
    double v0 = 0.05; // the self-propulsion speed
    double Dr = 1.0;  // the rotational diffusion constant

    //The defaults can be overridden from the command line
    while((c=getopt(argc,argv,"n:t:g:i:e:p:a:v:d:r:")) != -1)
        switch(c)
        {
            case 'n': numpts = atoi(optarg); break;
            case 't': tSteps = atoi(optarg); break;
            case 'g': USE_GPU = atoi(optarg); break;
            case 'i': initSteps = atoi(optarg); break;
            case 'e': dt = atof(optarg); break;
            case 'p': p0 = atof(optarg); break;
            case 'a': a0 = atof(optarg); break;
            case 'v': v0 = atof(optarg); break;
            case 'd': Dr = atof(optarg); break;
            case 'r': reproducible_flag = atoi(optarg); break;
            case '?':
                    if(optopt=='c')
                        std::cerr<<"Option -" << optopt << "requires an argument.\n";
                    else if(isprint(optopt))
                        std::cerr<<"Unknown option '-" << optopt << "'.\n";
                    else
                        std::cerr << "Unknown option character.\n";
                    return 1;
            default:
                       abort();
        };

    bool reproducible = (reproducible_flag == 1);

    clock_t t1,t2; //clocks for timing information
    //check to see if we should run on a GPU
    bool initializeGPU = true;
    bool gpu = chooseGPU(USE_GPU);
    if (!gpu)
        initializeGPU = false;

    //set-up a log-spaced state saver...can add as few as 1 database, or as many as you'd like. "0.1" will save 10 states per decade of time
    logEquilibrationStateWriter lewriter(0.01);
    char dataname[256];
    double equilibrationTime = dt*initSteps;
    vector<long long int> offsets;
    offsets.push_back(0);
    //offsets.push_back(100);offsets.push_back(1000);offsets.push_back(50);
    for(int ii = 0; ii < offsets.size(); ++ii)
        {
        if(reproducible)
            sprintf(dataname,"test_N%i_p%.3f_a_bimodal_v%.3f_Dr%.3f_dt%.4f_repro_et%.6f.nc",numpts,p0,v0,Dr,dt,offsets[ii]*dt);
        else
            {
            std::random_device rd;
            sprintf(dataname,"test_N%i_p%.3f_a_bimodal_v%.3f_Dr%.3f_dt%.4f_rand%u_et%.6f.nc",numpts,p0,v0,Dr,dt,rd(),offsets[ii]*dt);
            }
        shared_ptr<simpleVoronoiDatabase> ncdat=make_shared<simpleVoronoiDatabase>(numpts,dataname,fileMode::replace);
        lewriter.addDatabase(ncdat,offsets[ii]);
        }
    lewriter.identifyNextFrame();


    cout << "initializing a system of " << numpts << " self-propelled cells with v0 = " << v0 << ", Dr = " << Dr << endl;
    shared_ptr<selfPropelledParticleDynamics> spp = make_shared<selfPropelledParticleDynamics>(numpts);

    //define a voronoi configuration with a quadratic energy functional
    shared_ptr<VoronoiQuadraticEnergy> voronoiModel  = make_shared<VoronoiQuadraticEnergy>(numpts,1.0,4.0,reproducible,initializeGPU);

    //set the cell preferences to have a target p0, and a_0 chosen from a list
    voronoiModel->setCellPreferencesWithRandomAreaList(p0,{0.8, 1.2});

    //set the activity of the cells
    voronoiModel->setv0Dr(v0,Dr);

    //combine the equation of motion and the cell configuration in a "Simulation"
    SimulationPtr sim = make_shared<Simulation>();
    sim->setConfiguration(voronoiModel);
    sim->addUpdater(spp,voronoiModel);
    //set the time step size
    sim->setIntegrationTimestep(dt);
    //initialize Hilbert-curve sorting... can be turned off by commenting out this line or seting the argument to a negative number
    //sim->setSortPeriod(initSteps/10);
    //set appropriate CPU and GPU flags
    sim->setCPUOperation(!initializeGPU);
    if (!gpu)
        sim->setOmpThreads(abs(USE_GPU));
    sim->setReproducible(reproducible);

    //run for a few initialization timesteps
    printf("starting initialization\n");
    for(long long int ii = 0; ii < initSteps; ++ii)
        {
        sim->performTimestep();
        };
    voronoiModel->computeGeometry();
    printf("Finished with initialization\n");
    cout << "current q = " << voronoiModel->reportq() << endl;
    //the reporting of the force should yield a number that is numerically close to zero.
    voronoiModel->reportMeanCellForce(false);

    //run for additional timesteps, compute dynamical features, and record timing information
    dynamicalFeatures dynFeat(voronoiModel->returnPositions(),voronoiModel->Box);
    t1=clock();
//    cudaProfilerStart();
    for(long long int ii = 0; ii < tSteps; ++ii)
        {

        if (ii == lewriter.nextFrameToSave)
            {
            lewriter.writeState(voronoiModel,ii);
            }

        sim->performTimestep();
        };
//    cudaProfilerStop();
    t2=clock();
    printf("final state:\t\t energy %f \t msd %f \t overlap %f\n",voronoiModel->computeEnergy(),dynFeat.computeMSD(voronoiModel->returnPositions()),dynFeat.computeOverlapFunction(voronoiModel->returnPositions()));
    double steptime = (t2-t1)/(double)CLOCKS_PER_SEC/tSteps;
    cout << "timestep ~ " << steptime << " per frame; " << endl;

    if(initializeGPU)
        cudaDeviceReset();
    return 0;
};