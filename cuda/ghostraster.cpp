#include "ghostraster.h"
#include "../geometry.h"

int lis::GhostRaster::elements(Geometry& geometry)
{
	return (geometry.xsz+2)*(geometry.ysz+2);
}

int lis::GhostRaster::pitch(Geometry& geometry)
{
	return geometry.xsz + 2;
}

int lis::GhostRaster::offset(Geometry& geometry)
{
	return geometry.xsz + 3;
}

NUMERIC_TYPE* lis::GhostRaster::allocate(Geometry& geometry)
{
	return new NUMERIC_TYPE[elements(geometry)]();
}
