cmake_minimum_required(VERSION 3.10)
project(enmod_app LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
    src/Grid.cpp
    src/Logger.cpp
    src/ScenarioGenerator.cpp
    src/Solver.cpp
    src/Policy.cpp
    src/HtmlReportGenerator.cpp
    src/Cost.cpp
    src/BIDP.cpp
    src/FIDP.cpp
    src/AVI.cpp
    src/API.cpp
    src/PolicyGenerator.cpp
    src/PolicyVerifier.cpp
    src/DynamicBIDPSolver.cpp
    src/DynamicAPISolver.cpp
    src/DynamicFIDPSolver.cpp
    src/DynamicAVISolver.cpp
    src/RLSolver.cpp
    src/QLearningSolver.cpp
    src/DynamicQLearningSolver.cpp
    src/SARSASolver.cpp
    src/DynamicSARSASolver.cpp
    src/ActorCriticSolver.cpp
    src/DynamicActorCriticSolver.cpp
    src/HybridDPRLSolver.cpp
    src/AdaptiveCostSolver.cpp
    src/InterlacedSolver.cpp
    src/HierarchicalSolver.cpp
    src/PolicyBlendingSolver.cpp
    src/MultiAgentCPSController.cpp
    src/MultiAgentReportGenerator.cpp
    src/SimulatedSensorNetwork.cpp
    src/RealTimeSensorNetwork.cpp
    
    # Parallel Solvers
    src/ParallelSolverBase.cu
    src/ParallelBIDP.cpp
    src/ParallelStaticSolvers.cpp
    src/ParallelDynamicBIDPSolver.cpp
    src/ParallelDynamicSolvers.cpp
    src/ParallelHybridSolvers.cpp
)

add_executable(enmod_app ${SOURCES})

target_include_directories(enmod_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(enmod_app PRIVATE CUDA::cudart)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})