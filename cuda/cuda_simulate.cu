#include "cuda_simulate.cuh"
#include "cuda_util.cuh"
#include "cuda_unifiedallocator.cuh"
#include "io.h"

void lis::cuda::Simulation::print_device_info()
{
	int device = cuda::get_device();
	cudaDeviceProp cudaDevProp;
	cuda::get_device_properties(cudaDevProp, device);
	printf("Using CUDA device: %i\t", device);
	unsigned char* bytes = reinterpret_cast<unsigned char*>(cudaDevProp.uuid.bytes);
	printf("UUID: GPU-%02x%02x%02x%02x-", bytes[0], bytes[1], bytes[2], bytes[3]);
	printf("%02x%02x-", bytes[4], bytes[5]);
	printf("%02x%02x-", bytes[6], bytes[7]);
	printf("%02x%02x-", bytes[8], bytes[9]);
	printf("%02x%02x%02x%02x%02x%02x\n", bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

void lis::cuda::Simulation::initialise_H
(
	NUMERIC_TYPE* H,
	const char* filename,
	States& states,
	NUMERIC_TYPE* DEM,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose
)
{
	if (states.startfile == ON)
	{
		StartFile::load(filename, H, geometry, pitch, offset, verbose);
		if (states.startelev == ON)
		{
			StartFile::subtract_dem(H, DEM, geometry, pitch, offset);
		}
	}
}

void lis::cuda::Simulation::initialise_discharge
(
	NUMERIC_TYPE* HU,
	NUMERIC_TYPE* HV,
	const char* startfilename,
	States& states,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose
)
{
	if (states.startq2d == ON)
	{
		char hu_startfile[800];
		strcpy(hu_startfile, startfilename);
		strcat(hu_startfile, ".Qx");

		char hv_startfile[800];
		strcpy(hv_startfile, startfilename);
		strcat(hv_startfile, ".Qy");

		StartFile::load(hu_startfile, HU, geometry, pitch, offset, verbose);
		StartFile::load(hv_startfile, HV, geometry, pitch, offset, verbose);
	}
}

void lis::cuda::Simulation::initialise_manning
(
	NUMERIC_TYPE* manning,
	const char* filename,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose
)
{
	FILE* file = fopen_or_die(filename, "rb", "Loading manningfile\n", verbose);
	Geometry manning_file_geometry;
	NUMERIC_TYPE no_data_value;
	AsciiRaster::read_header(file, manning_file_geometry, no_data_value);
	AsciiRaster::match_cell_dimensions_or_die(geometry, manning_file_geometry,
			"initialise_manning");
	AsciiRaster::read(file, manning, geometry, pitch, offset);
	fclose(file);
}

void lis::cuda::Simulation::load_boundaries
(
	Fnames& filenames,
	States& states,
	Pars& pars,
	BoundCs& boundCs,
	int verbose
)
{
	LoadBCs(&filenames, &states, &pars, &boundCs, verbose);
	LoadBCVar(&filenames, &states, &pars, &boundCs, nullptr, nullptr, nullptr,
			verbose);
}

void lis::cuda::Simulation::update_geometry
(
	Pars& dst,
	Geometry& src
)
{
	dst.xsz = src.xsz;
	dst.ysz = src.ysz;
	dst.blx = src.blx;
	dst.bly = src.bly;
	dst.tly = src.tly;
	dst.dx = src.dx;
	dst.dy = src.dy;
}
