#include "cuda_fv1_snapshot.cuh"
#include "io.h"

lis::cuda::fv1::Snapshot::Snapshot
(
	const char* resrootname,
	NUMERIC_TYPE interval,
	NUMERIC_TYPE next_save,
	int counter,
	NUMERIC_TYPE* DEM,
	Flow& U,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose,
	int precision
)
:
cuda::Snapshot<Flow>(resrootname, interval, next_save, counter, U, geometry,
		pitch, offset, verbose, precision),
DEM(DEM),
writer_elev(std::async(std::launch::async, []{})),
writer_H(std::async(std::launch::async, []{})),
writer_HU(std::async(std::launch::async, []{})),
writer_HV(std::async(std::launch::async, []{})),
writer_U(std::async(std::launch::async, []{})),
writer_V(std::async(std::launch::async, []{}))
{}

void lis::cuda::fv1::Snapshot::write()
{
	writer_H = write_async(U.H, ".wd");

	if (write_elevation)
	{
		writer_elev = write_async(ElevationWriter(U.H, DEM), ".elev");
	}

	if (write_discharge)
	{
		writer_HU = write_async(U.HU, ".Qx");
		writer_HV = write_async(U.HV, ".Qy");
	}

	if (write_velocity)
	{
		writer_U = write_async(VelocityWriter(U.H, U.HU, DepthThresh), ".Vx");
		writer_V = write_async(VelocityWriter(U.H, U.HV, DepthThresh), ".Vy");
	}
}

void lis::cuda::fv1::Snapshot::wait()
{
	writer_elev.wait();
	writer_H.wait();
	writer_HU.wait();
	writer_HV.wait();
	writer_U.wait();
	writer_V.wait();
}
