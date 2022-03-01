#pragma once
#include "../lisflood.h"
#include "../geometry.h"
#include "../rain/rain.h"

namespace lis
{
namespace cuda
{

class Simulation
{
protected:
	void print_device_info();

	void initialise_H
	(
		NUMERIC_TYPE* H,
		const char* filename,
		States& states,
		NUMERIC_TYPE* DEM,
		Geometry& geometry,
		int pitch,
		int offset,
		int verbose
	);

	void initialise_discharge
	(
		NUMERIC_TYPE* HU,
		NUMERIC_TYPE* HV,
		const char* filename,
		States& states,
		Geometry& geometry,
		int pitch,
		int offset,
		int verbose
	);

	void initialise_manning
	(
		NUMERIC_TYPE* manning,
		const char* filename,
		Geometry& geometry,
		int pitch,
		int offset,
		int verbose
	);

	void update_geometry
	(
		Pars& dst,
		Geometry& src
	);

	void load_boundaries
	(
		Fnames& filenames,
		States& states,
		Pars& pars,
		BoundCs& boundCs,
		int verbose
	);
};

}
}
