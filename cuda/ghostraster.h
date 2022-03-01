#pragma once
#include "../lisflood.h"
#include "../geometry.h"

namespace lis
{

struct GhostRaster
{
	static int elements(Geometry& geometry);
	static int pitch(Geometry& geometry);
	static int offset(Geometry& geometry);
	static NUMERIC_TYPE* allocate(Geometry& geometry);
};

}

