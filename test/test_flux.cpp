#include "catch.hpp"
#include "checks.h"
#include "../swe/flux.h"

TEST_CASE("physical_flux_above_DepthThresh")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->DepthThresh = 1e-6;
	Solverptr->g = C(9.8);

	NUMERIC_TYPE H = C(1.5);
	NUMERIC_TYPE HU = C(2.0);
	NUMERIC_TYPE HV = C(3.0);

	NUMERIC_TYPE fH = C(0.0);
	NUMERIC_TYPE fHU = C(0.0);
	NUMERIC_TYPE fHV = C(0.0);

	physical_flux(Solverptr, H, HU, HV, fH, fHU, fHV);

	CHECK( fH == approx(C(2.0)) );
	CHECK( fHU == approx(C(13.6917)) );
	CHECK( fHV == approx(C(4.0)) );
}

TEST_CASE("physical_flux_below_DepthThresh_is_zero")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->DepthThresh = 1e-6;
	Solverptr->g = C(9.8);

	NUMERIC_TYPE H = C(1e-7);
	NUMERIC_TYPE HU = C(2.0);
	NUMERIC_TYPE HV = C(3.0);

	NUMERIC_TYPE fH = C(1.0);
	NUMERIC_TYPE fHU = C(1.0);
	NUMERIC_TYPE fHV = C(1.0);

	physical_flux(Solverptr, H, HU, HV, fH, fHU, fHV);

	CHECK( fH == approx(C(0.0)) );
	CHECK( fHU == approx(C(0.0)) );
	CHECK( fHV == approx(C(0.0)) );
}
