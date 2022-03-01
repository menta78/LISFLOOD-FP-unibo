#include "ghostraster.h"
#include "cuda_geometry.cuh"
#include "cuda_util.cuh"

NUMERIC_TYPE* lis::cuda::GhostRaster::allocate_pinned
(
	Geometry& geometry
)
{
	return static_cast<NUMERIC_TYPE*>(malloc_pinned(
				lis::GhostRaster::elements(geometry)*sizeof(NUMERIC_TYPE)));
}

NUMERIC_TYPE* lis::cuda::GhostRaster::allocate_device
(
	Geometry& geometry
)
{
	return static_cast<NUMERIC_TYPE*>(malloc_device(
				lis::GhostRaster::elements(geometry)*sizeof(NUMERIC_TYPE)));
}

void lis::cuda::GhostRaster::copy
(
	NUMERIC_TYPE* dst,
	NUMERIC_TYPE* src,
	Geometry& geometry
)
{
	checkCudaErrors(cudaMemcpy(dst, src,
				lis::GhostRaster::elements(geometry)*sizeof(NUMERIC_TYPE),
				cudaMemcpyDefault));
}
