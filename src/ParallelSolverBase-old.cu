#include "enmod/ParallelSolverBase.h"
#include "enmod/Logger.h"
#include "enmod/Cost.h" // For Cost::current_mode
#include "enmod/HtmlReportGenerator.h"
#include <cuda_runtime.h>
#include <string>
#include <iostream>

// --- DEVICE HELPERS ---

__device__ const DeviceCost D_MAX_COST = { DEVICE_MAX_INT, DEVICE_MAX_INT, DEVICE_MAX_INT };

__device__ __forceinline__ bool d_cost_less(const DeviceCost& a, const DeviceCost& b, DeviceEvacuationMode mode) {
    switch (mode) {
        case DeviceEvacuationMode::PANIC:
        case DeviceEvacuationMode::ALERT:
            if (a.smoke != b.smoke) return a.smoke < b.smoke;
            if (a.time != b.time) return a.time < b.time;
            return a.distance < b.distance;
        case DeviceEvacuationMode::NORMAL:
        default:
            if (a.time != b.time) return a.time < b.time;
            if (a.distance != b.distance) return a.distance < b.distance;
            return a.smoke < b.smoke;
    }
}

__device__ __forceinline__ DeviceCost d_cost_add(const DeviceCost& a, const DeviceCost& b) {
    if (a.distance == DEVICE_MAX_INT || b.distance == DEVICE_MAX_INT) return D_MAX_COST;
    return {a.smoke + b.smoke, a.time + b.time, a.distance + b.distance};
}

// --- CUDA KERNEL ---

__global__ void relaxation_kernel(const DeviceGridCell* d_grid, const DeviceCost* d_in, DeviceCost* d_out, int* d_changed, int rows, int cols, DeviceEvacuationMode mode) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    int idx = r * cols + c;
    DeviceGridCell cell = d_grid[idx];
    DeviceCost current_val = d_in[idx];

    if (!cell.is_walkable) {
        d_out[idx] = current_val; 
        return;
    }
    
    // If it's a source node (cost 0), it stays 0.
    if (current_val.distance == 0 && current_val.time == 0 && current_val.smoke == 0) {
        d_out[idx] = current_val;
        return;
    }

    DeviceCost best_neighbor = current_val;
    DeviceCost move_cost = cell.move_cost;

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    for(int i=0; i<4; ++i) {
        int nr = r + dr[i];
        int nc = c + dc[i];
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
            DeviceCost n_cost = d_in[nr * cols + nc];
            if (n_cost.distance != DEVICE_MAX_INT) {
                DeviceCost path_cost = d_cost_add(n_cost, move_cost);
                if (d_cost_less(path_cost, best_neighbor, mode)) {
                    best_neighbor = path_cost;
                }
            }
        }
    }

    d_out[idx] = best_neighbor;
    if (d_cost_less(best_neighbor, current_val, mode)) {
        atomicOr(d_changed, 1);
    }
}

// --- HOST CLASS IMPLEMENTATION ---

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::string s = std::string(msg) + ": " + cudaGetErrorString(err);
        Logger::log(LogLevel::ERROR, s);
        throw std::runtime_error(s);
    }
}

ParallelSolverBase::ParallelSolverBase(const Grid& grid_ref, const std::string& name) 
    : Solver(grid_ref, name) {
    cost_map.assign(grid.getRows(), std::vector<Cost>(grid.getCols()));
    initialize_gpu_memory();
    upload_grid_data();
}

ParallelSolverBase::~ParallelSolverBase() {
    free_gpu_memory();
}

void ParallelSolverBase::initialize_gpu_memory() {
    size_t n_cells = grid.getRows() * grid.getCols();
    checkCuda(cudaMalloc(&d_grid, n_cells * sizeof(DeviceGridCell)), "Malloc d_grid");
    checkCuda(cudaMalloc(&d_cost_in, n_cells * sizeof(DeviceCost)), "Malloc d_cost_in");
    checkCuda(cudaMalloc(&d_cost_out, n_cells * sizeof(DeviceCost)), "Malloc d_cost_out");
    checkCuda(cudaMalloc(&d_changed_flag, sizeof(int)), "Malloc d_changed");
}

void ParallelSolverBase::free_gpu_memory() {
    if(d_grid) cudaFree(d_grid);
    if(d_cost_in) cudaFree(d_cost_in);
    if(d_cost_out) cudaFree(d_cost_out);
    if(d_changed_flag) cudaFree(d_changed_flag);
}

void ParallelSolverBase::upload_grid_data() {
    int rows = grid.getRows();
    int cols = grid.getCols();
    std::vector<DeviceGridCell> h_grid(rows * cols);
    
    for(int r=0; r<rows; ++r) {
        for(int c=0; c<cols; ++c) {
            int idx = r * cols + c;
            h_grid[idx].is_walkable = grid.isWalkable(r, c);
            h_grid[idx].is_exit = grid.isExit(r, c); 
            Cost mc = grid.getMoveCost({r, c});
            h_grid[idx].move_cost = {mc.smoke, mc.time, mc.distance};
        }
    }
    checkCuda(cudaMemcpy(d_grid, h_grid.data(), h_grid.size() * sizeof(DeviceGridCell), cudaMemcpyHostToDevice), "Copy Grid");
}

void ParallelSolverBase::reset_cost_map(const std::vector<Position>& source_nodes) {
    int rows = grid.getRows();
    int cols = grid.getCols();
    std::vector<DeviceCost> h_costs(rows * cols, {DEVICE_MAX_INT, DEVICE_MAX_INT, DEVICE_MAX_INT});

    for (const auto& p : source_nodes) {
        if (grid.isValid(p.row, p.col)) {
            h_costs[p.row * cols + p.col] = {0, 0, 0};
        }
    }
    checkCuda(cudaMemcpy(d_cost_in, h_costs.data(), h_costs.size() * sizeof(DeviceCost), cudaMemcpyHostToDevice), "Copy Init Costs");
}

void ParallelSolverBase::run_cuda_algo(const std::vector<Position>& source_nodes) {
    reset_cost_map(source_nodes);

    int rows = grid.getRows();
    int cols = grid.getCols();
    int changed_h = 1;
    int max_iter = 2 * rows * cols;
    int iter = 0;

    DeviceEvacuationMode mode = static_cast<DeviceEvacuationMode>(Cost::current_mode);

    dim3 block(16, 16);
    dim3 grid_dim((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    while (changed_h && iter < max_iter) {
        changed_h = 0;
        checkCuda(cudaMemcpy(d_changed_flag, &changed_h, sizeof(int), cudaMemcpyHostToDevice), "Reset Flag");

        relaxation_kernel<<<grid_dim, block>>>(d_grid, d_cost_in, d_cost_out, d_changed_flag, rows, cols, mode);
        checkCuda(cudaDeviceSynchronize(), "Kernel Sync");

        checkCuda(cudaMemcpy(&changed_h, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost), "Read Flag");
        std::swap(d_cost_in, d_cost_out); // Pointer swap on host
        iter++;
    }
    download_results();
}

void ParallelSolverBase::download_results() {
    int rows = grid.getRows();
    int cols = grid.getCols();
    std::vector<DeviceCost> h_costs(rows * cols);
    // d_cost_in holds the latest valid data after swap
    checkCuda(cudaMemcpy(h_costs.data(), d_cost_in, h_costs.size() * sizeof(DeviceCost), cudaMemcpyDeviceToHost), "Download Results");

    for(int r=0; r<rows; ++r) {
        for(int c=0; c<cols; ++c) {
            DeviceCost dc = h_costs[r * cols + c];
            if (dc.distance == DEVICE_MAX_INT) cost_map[r][c] = {};
            else cost_map[r][c] = {dc.smoke, dc.time, dc.distance};
        }
    }
}

const std::vector<std::vector<Cost>>& ParallelSolverBase::getCostMap() const { return cost_map; }
Cost ParallelSolverBase::getEvacuationCost() const { 
    auto start = grid.getStartPosition();
    if(grid.isValid(start.row, start.col)) return cost_map[start.row][start.col];
    return {}; 
}
void ParallelSolverBase::generateReport(std::ofstream& report_file) const {
    report_file << "<h2>Final Cost Map (CUDA Accelerated)</h2>\n";
    report_file << grid.toHtmlStringWithCost(cost_map);
}