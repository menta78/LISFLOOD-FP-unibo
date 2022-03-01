#pragma once
#include "../lisflood.h"
#include "../geometry.h"

namespace lis
{
namespace cuda
{

struct GhostRaster
{
	static NUMERIC_TYPE* allocate_pinned
	(
		Geometry& geometry
	);

	static NUMERIC_TYPE* allocate_device
	(
		Geometry& geometry
	);

	static void copy
	(
		NUMERIC_TYPE* dst,
		NUMERIC_TYPE* src,
		Geometry& geometry
	);
};

}
}
