# CellGPU: Project Implementation, Models, and Usage Guide

This document provides a comprehensive overview of the CellGPU project, detailing its core scientific models, software implementation, and instructions for building and running simulations.

## 1. Overview

**CellGPU** is a high-performance simulation package designed for studying 2D off-lattice models of cellular tissues. It implements GPU-accelerated algorithms for two primary model types:

1.  **The Voronoi Model:** Cells are defined by the Voronoi tessellation of a set of points (cell centers). The model simulates the dynamics of these cell centers.
2.  **The Vertex Model:** Cells are represented as polygons, and the degrees of freedom are the positions of the vertices that form the cellular network.

The framework is built with C++ and CUDA, and it relies on several external libraries for geometry (CGAL), linear algebra (Eigen), and data storage (HDF5/netCDF, though this is being phased out).

## 2. Core Scientific Models

CellGPU is more than a standard molecular dynamics (MD) package; a key feature is the enforcement of topological rules during each timestep.

### 2.1. 2D Voronoi Model

In this model, the topology of the cellular monolayer is defined by the Delaunay triangulation of the cell center positions. The simulation follows a hybrid CPU/GPU operational flow:

1.  **(GPU/CPU) Triangulation:** A Delaunay triangulation of the cell positions is performed. This can be done using the custom `DelaunayGPU` engine or a `DelaunayCGAL` wrapper.
2.  **(GPU) Integration:** Cell positions are updated by integrating equations of motion, with forces computed on the GPU.
3.  **(GPU) Topology Check:** The validity of the previous Delaunay triangulation is checked against the new cell positions. Any triangle whose circumcircle now contains another cell's center is flagged.
4.  **(GPU/CPU) Topology Repair:** For any flagged cells, the neighbor list is repaired to reflect the new topology.
5.  **(GPU/CPU) Update & Repeat:** Data structures are updated with the new topology, and the cycle repeats.

### 2.2. 2D Vertex Model

In the Active Vertex Model (AVM), cells are explicit polygons, and the simulation evolves the positions of their shared vertices. The simulation loop is simpler and can run almost entirely on the GPU:

1.  **(CPU) Initialization:** A starting network of polygonal cells is created, typically from a triangulation of random points.
2.  **(GPU) Geometry Calculation:** The geometric properties of each cell (area, perimeter) are computed from the vertex positions.
3.  **(GPU) Force Calculation:** Forces on each vertex are calculated based on an energy functional, which typically penalizes deviations from a preferred area and perimeter.
4.  **(GPU) Integration:** Vertex positions are updated based on the forces and any active dynamics.
5.  **(GPU) Topology Check (T1 transitions):** The code checks for edges that have shrunk below a certain threshold, triggering a "T1" topological transition (an edge flip) to maintain the network structure. Data structures are updated on the GPU, and the cycle repeats.

## 3. Implementation Details

### 3.1. Technology Stack

-   **Language:** C++ (requires C++14, moving towards C++17)
-   **GPU Computing:** CUDA (tested with v11.0, requires compute capability 3.5+)
-   **Build System:** CMake
-   **Core Dependencies:**
    -   **CGAL:** For geometric algorithms, particularly triangulation.
    -   **Eigen:** For matrix operations (e.g., diagonalizing the dynamical matrix).
    -   **gmp & mpfr:** Required by CGAL.
    -   **HDF5/netCDF:** Used by some database classes for trajectory storage.

### 3.2. Code Architecture

The source code in the `/src` directory is organized modularly:

-   `models`: Contains the core logic for the Voronoi and Vertex models, including force calculations (`VertexQuadraticEnergy`) and topology maintenance (`voronoiModelBase`).
-   `updaters`: Implements equations of motion, such as Brownian dynamics (`brownianParticleDynamics`), active particle dynamics (`selfPropelledParticleDynamics`), and energy minimization (`EnergyMinimizerFIRE`).
-   `utilities`: Provides helper classes like `GPUArray` for data management, spatial sorting algorithms, and random number generators.
-   `databases`: Defines classes for saving simulation data to different file formats (e.g., `DatabaseNetCDFAVM`, `DatabaseTextVoronoi`).
-   `analysis`: Contains tools for on-the-fly data analysis, such as autocorrelation functions.

The `Simulation` class acts as a central orchestrator, tying together the model (configuration), updater (EOM), and other components. Data is primarily stored in `GPUArray` objects, a structure inspired by HOOMD-blue that manages data allocation and transfers between the CPU (host) and GPU (device).

## 4. Building and Running Simulations

### 4.1. Dependencies and Compilation

First, ensure all dependencies (CUDA, CGAL, Eigen, etc.) are installed and available in your system's path. On Ubuntu, most can be installed via `apt-get`. The build process uses CMake and is straightforward:

```bash
# Navigate to the build directory
$ cd build/

# Configure the project
$ cmake ..

# Compile the executables
$ make
```

By default, this compiles executables from `voronoi.cpp` and `Vertex.cpp`. To compile other examples, you must add the file's base name to the `CMakeLists.txt` file.

### 4.2. Running a Simulation

Simulations are launched by running the compiled executables with command-line arguments. The example file `examples/cellDivision.cpp` provides a clear template for how simulations are configured and run.

When compiled, it can be run to simulate either a Voronoi or Vertex model with cell division. The behavior is controlled by the following command-line flags:

| Flag | Argument | Description |
| :--- | :--- | :--- |
| `-n` | `int` | Number of initial cells. |
| `-t` | `int` | Number of timesteps for the production run. |
| `-i` | `int` | Number of timesteps for initialization/equilibration. |
| `-g` | `int` | Index of the GPU to use. A negative value (e.g., -1) forces CPU-only operation. |
| `-z` | `int` | **Program Switch:** Selects the model. `>= 0` for Voronoi, `< 0` for Vertex. |
| `-e` | `double` | Timestep size (`dt`). |
| `-p` | `double` | Target/preferred perimeter (`p_0`). |
| `-a` | `double` | Target/preferred area (`a_0`). |
| `-v` | `double` | Self-propulsion speed (`v_0`) for active models. |
| `-d` | `double` | Rotational diffusion rate (`Dr`) for active models. |
| `-s` | `double` | Surface tension parameter (`gamma`) for Voronoi models with tension. |

**Example Command (Vertex Model):**
```bash
# Run a vertex model simulation with 200 cells on GPU 0
./cellDivision -n 200 -g 0 -z -1 -p 3.8 -v 0.01 -t 1000
```

**Example Command (Voronoi Model):**
```bash
# Run a Voronoi model simulation with 200 cells on the CPU
./cellDivision -n 200 -g -1 -z 0 -p 3.84 -v 0.01 -t 1000
```

### 4.3. Included Examples

The repository includes several C++ files that demonstrate different features of the framework. These serve as excellent starting points for building your own simulations.

*   `voronoi.cpp` / `Vertex.cpp`: Basic simulations of the two main models.
*   `minimize.cpp`: Demonstrates energy minimization using the FIRE algorithm.
*   `tensions.cpp` / `vertexTensions.cpp`: Shows how to add line tension terms.
*   `cellDivision.cpp` / `cellDeath.cpp`: Implements simulations where the number of cells changes.
*   `dynMat.cpp`: Computes and diagonalizes the dynamical matrix of a Voronoi model.
*   `nvtVoronoi.cpp`: Sets up a simulation in the NVT ensemble using a Nosé-Hoover thermostat.

**Note:** The documentation warns that the examples in the `examples/` directory may not have been actively maintained and might require some adjustments to compile and run with the latest code.