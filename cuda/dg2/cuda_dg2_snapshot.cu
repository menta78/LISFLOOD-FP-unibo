#include "cuda_dg2_snapshot.cuh"

lis::cuda::dg2::Snapshot::Snapshot
(
	const char* resrootname,
	NUMERIC_TYPE interval,
	NUMERIC_TYPE next_save,
	int counter,
	Topography& DEM,
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
U(U),
writer_elev(std::async(std::launch::async, []{})),
writer_H(std::async(std::launch::async, []{})),
writer_H1x(std::async(std::launch::async, []{})),
writer_H1y(std::async(std::launch::async, []{})),
writer_HU(std::async(std::launch::async, []{})),
writer_HU1x(std::async(std::launch::async, []{})),
writer_HU1y(std::async(std::launch::async, []{})),
writer_HV(std::async(std::launch::async, []{})),
writer_HV1x(std::async(std::launch::async, []{})),
writer_HV1y(std::async(std::launch::async, []{})),
writer_U(std::async(std::launch::async, []{})),
writer_V(std::async(std::launch::async, []{}))
{}

void lis::cuda::dg2::Snapshot::write()
{
	writer_H = write_async(U.H, ".wd");
	writer_H1x = write_async(U.H1x, ".wd1x");
	writer_H1y = write_async(U.H1y, ".wd1y");

	if (write_elevation)
	{
		writer_elev = write_async(ElevationWriter(U.H, DEM._0), ".elev");
	}

	if (write_discharge)
	{
		writer_HU = write_async(U.HU, ".Qx");
		writer_HU1x = write_async(U.HU1x, ".Qx1x");
		writer_HU1y = write_async(U.HU1y, ".Qx1y");
		writer_HV = write_async(U.HV, ".Qy");
		writer_HV1x = write_async(U.HV1x, ".Qy1x");
		writer_HV1y = write_async(U.HV1y, ".Qy1y");
	}

	if (write_velocity)
	{
		writer_U = write_async(VelocityWriter(U.H, U.HU, DepthThresh), ".Vx");
		writer_V = write_async(VelocityWriter(U.H, U.HV, DepthThresh), ".Vy");
	}
}

void lis::cuda::dg2::Snapshot::wait()
{
	writer_elev.wait();
	writer_H.wait();
	writer_H1x.wait();
	writer_H1y.wait();
	writer_HU.wait();
	writer_HU1x.wait();
	writer_HU1y.wait();
	writer_HV.wait();
	writer_HV1x.wait();
	writer_HV1y.wait();
	writer_U.wait();
	writer_V.wait();
}
