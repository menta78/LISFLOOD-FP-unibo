#include "cuda_max_field.cuh"
#include "cuda_solver.cuh"
#include "io.h"

namespace lis
{
namespace cuda
{

__global__ void
__launch_bounds__(CUDA_BLOCK_SIZE)
update_max_field
(
	NUMERIC_TYPE* maxH,
	NUMERIC_TYPE* H
)
{
	int global_i = blockIdx.x*blockDim.x + threadIdx.x;
	int global_j = blockIdx.y*blockDim.y + threadIdx.y;

	for (int j=global_j+1; j<cuda::geometry.ysz+1; j+=blockDim.y*gridDim.y)
	{
		for (int i=global_i+1; i<cuda::geometry.xsz+1; i+=blockDim.x*gridDim.x)
		{
			NUMERIC_TYPE& maxHvalue = maxH[j*cuda::pitch+i];
			maxHvalue = FMAX(maxHvalue, H[j*cuda::pitch+i]);
		}
	}
}
}
}

lis::cuda::MaxField::MaxField
(
	const char* resrootname,
	Geometry& geometry,
	int pitch,
	int offset,
	dim3 grid_size,
	int precision
)
:
resrootname(resrootname),
geometry(geometry),
pitch(pitch),
offset(offset),
grid_size(grid_size),
precision(precision)
{
	maxH = GhostRaster::allocate_pinned(geometry);
}

void lis::cuda::MaxField::update
(
	NUMERIC_TYPE* H
)
{
	lis::cuda::update_max_field<<<grid_size, CUDA_BLOCK_SIZE>>>(maxH, H);
}

void lis::cuda::MaxField::write()
{
	char filename[800];
	snprintf(filename, 800*sizeof(char), "%s%s", resrootname, ".max");

	FILE* file = fopen_or_die(filename, "wb");
	
	AsciiRaster::write(file, maxH, geometry, pitch, offset, NULLVAL,
			precision);

	fclose(file);
}

lis::cuda::MaxField::~MaxField()
{
	free_pinned(maxH);
}
