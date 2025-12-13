#ifndef ENMOD_PARALLEL_SOLVER_BASE_H
#define ENMOD_PARALLEL_SOLVER_BASE_H

#include "Solver.h"
#include "CudaTypes.h"
#include <vector>

class ParallelSolverBase : public Solver {
public:
    ParallelSolverBase(const Grid& grid_ref, const std::string& name);
    virtual ~ParallelSolverBase();

    // The core driver that children will call with specific source nodes
    void run_cuda_algo(const std::vector<Position>& source_nodes);

    const std::vector<std::vector<Cost>>& getCostMap() const;
    Cost getEvacuationCost() const override; 
    void generateReport(std::ofstream& report_file) const override;

protected:
    std::vector<std::vector<Cost>> cost_map;

private:
    void initialize_gpu_memory();
    void upload_grid_data();
    void reset_cost_map(const std::vector<Position>& source_nodes);
    void download_results();
    void free_gpu_memory();

    DeviceGridCell* d_grid = nullptr;
    DeviceCost* d_cost_in = nullptr;
    DeviceCost* d_cost_out = nullptr;
    int* d_changed_flag = nullptr;
};

#endif // ENMOD_PARALLEL_SOLVER_BASE_H