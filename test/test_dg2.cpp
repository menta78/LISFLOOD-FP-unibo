#include "catch.hpp"
#include "checks.h"
#include "../utility.h"
#include "../swe/dg2/fields.h"
#include "../swe/dg2/modifiedvars.h"
#include "../swe/fields.h"

using namespace dg2;

TEST_CASE("interface_limits")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 0;
	Parptr->ysz = 0;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	allocate_fields(Parptr, Arrptr);

	Arrptr->H[0*Parptr->xsz + 0] = C(3.5);
	Arrptr->H1x[0*Parptr->xsz + 0] = C(0.5)/SQRT(C(3.0));
	Arrptr->H1y[0*Parptr->xsz + 0] = C(1.0)/SQRT(C(3.0));

	CHECK( limit_neg(Arrptr->H, Arrptr->H1x, Parptr, 0, 0) == approx(4.0) );
	CHECK( limit_neg(Arrptr->H, Arrptr->H1y, Parptr, 0, 0) == approx(4.5) );
	CHECK( limit_pos(Arrptr->H, Arrptr->H1x, Parptr, 0, 0) == approx(3.0) );
	CHECK( limit_pos(Arrptr->H, Arrptr->H1y, Parptr, 0, 0) == approx(2.5) );
	
	memory_free_legacy(&Arrptr->H);
	deallocate_fields(Arrptr);
}

TEST_CASE("gauss_points")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 0;
	Parptr->ysz = 0;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	allocate_fields(Parptr, Arrptr);

	Arrptr->H[0*Parptr->xsz + 0] = C(3.5);
	Arrptr->H1x[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->H1y[0*Parptr->xsz + 0] = C(1.0);

	CHECK( gauss_lower(Arrptr->H, Arrptr->H1x, Parptr, 0, 0) == approx(3.0) );
	CHECK( gauss_lower(Arrptr->H, Arrptr->H1y, Parptr, 0, 0) == approx(2.5) );
	CHECK( gauss_upper(Arrptr->H, Arrptr->H1x, Parptr, 0, 0) == approx(4.0) );
	CHECK( gauss_upper(Arrptr->H, Arrptr->H1y, Parptr, 0, 0) == approx(4.5) );
	
	memory_free_legacy(&Arrptr->H);
	deallocate_fields(Arrptr);
}

TEST_CASE("update_HUstar")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 1;
	Parptr->ysz = 1;

	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->DepthThresh = C(1e-3);

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	allocate_fields(Parptr, Arrptr);

	Arrptr->H[0*Parptr->xsz + 0] = C(1.5);
	Arrptr->HU[0*Parptr->xsz + 0] = C(4.0);
	Arrptr->HU1x[0*Parptr->xsz + 0] = C(0.5)/SQRT(C(3.0));
	Arrptr->HU1y[0*Parptr->xsz + 0] = C(1.0)/SQRT(C(3.0));
	Arrptr->H1x[0*Parptr->xsz + 0] = C(1.5)/SQRT(C(3.0));
	Arrptr->H1y[0*Parptr->xsz + 0] = C(-1.0)/SQRT(C(3.0));
	Arrptr->Hstar_neg_x[0*Parptr->xsz + 0] = C(1.6);
	Arrptr->Hstar_pos_x[0*Parptr->xsz + 0] = C(10.0);
	Arrptr->Hstar_neg_y[0*Parptr->xsz + 0] = C(0.8);
	Arrptr->Hstar_pos_y[0*Parptr->xsz + 0] = C(1.5);

	FlowCoefficients U;
	U.H = Arrptr->H;
	U.H1x = Arrptr->H1x;
	U.H1y = Arrptr->H1y;
	U.HU = Arrptr->HU;
	U.HU1x = Arrptr->HU1x;
	U.HU1y = Arrptr->HU1y;

	update_HUstar(Parptr, Solverptr, Arrptr, U);

	CHECK( Arrptr->HUstar_neg_x[0*Parptr->xsz + 0] == approx(2.4) );
	CHECK( Arrptr->HUstar_pos_x[0*Parptr->xsz + 0] == approx(0.0) );
	CHECK( Arrptr->HUstar_neg_y[0*Parptr->xsz + 0] == approx(8.0) );
	CHECK( Arrptr->HUstar_pos_y[0*Parptr->xsz + 0] == approx(1.8) );
	
	memory_free_legacy(&Arrptr->H);
	deallocate_fields(Arrptr);
}
