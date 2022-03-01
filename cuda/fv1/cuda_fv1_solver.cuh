#pragma once
#include "../lisflood.h"
#include "params.h"
#include "stats.h"
#include "fv1/cuda_fv1_flow.cuh"
#include "cuda_sample.cuh"
#include "cuda_solver.cuh"

namespace lis
{
namespace cuda
{
namespace fv1
{

class Solver : public cuda::Solver<Flow>
{
public:
	Solver
	(
		Flow& U,
		NUMERIC_TYPE* DEM,
		NUMERIC_TYPE* Zstar_x,
		NUMERIC_TYPE* Zstar_y,
		NUMERIC_TYPE* manning,
		Geometry& geometry,
		PhysicalParams& physical_params,
		dim3 grid_size
	);

	void zero_ghost_cells();

	void update_ghost_cells();

	Flow& update_flow_variables
	(
		MassStats* mass_stats
	);

	Flow& d_U();

	void update_dt_per_element
	(
		NUMERIC_TYPE* dt_field
	) const;

	~Solver();

private:
	Flow U1;
	Flow U2;
	Flow Ux;
	Flow& Uold;
	Flow& U;
	NUMERIC_TYPE* DEM;
	NUMERIC_TYPE* Zstar_x;
	NUMERIC_TYPE* Zstar_y;
	NUMERIC_TYPE* manning;
	bool friction;
	const dim3 grid_size;
};

}
}
}
