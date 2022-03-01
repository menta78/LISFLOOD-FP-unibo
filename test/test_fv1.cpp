#include "catch.hpp"
#include "checks.h"
#include "../lisflood.h"
#include "../utility.h"
#include "../swe/fv1.h"
#include "../swe/fields.h"
#include "../swe/fv1/modifiedvars.h"

using namespace fv1;

TEST_CASE("Zstar_is_max_of_neighbouring_limits")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 2;
	Parptr->ysz = 2;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->DEM = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->DEM[0*Parptr->xsz + 0] = C(1.0);
	Arrptr->DEM[0*Parptr->xsz + 1] = C(4.0);
	Arrptr->DEM[1*Parptr->xsz + 0] = C(3.0);
	Arrptr->DEM[1*Parptr->xsz + 1] = C(2.0);
	
	allocate_swe_fields(Parptr, Arrptr);
	initialise_Zstar(Parptr, Arrptr);

	// internal faces
	CHECK( Arrptr->Zstar_x[0*(Parptr->xsz+1) + 1] == approx(C(4.0)) );
	CHECK( Arrptr->Zstar_x[1*(Parptr->xsz+1) + 1] == approx(C(3.0)) );

	CHECK( Arrptr->Zstar_y[1*(Parptr->xsz+1) + 0] == approx(C(3.0)) );
	CHECK( Arrptr->Zstar_y[1*(Parptr->xsz+1) + 1] == approx(C(4.0)) );

	// boundary faces
	CHECK( Arrptr->Zstar_x[0*(Parptr->xsz+1) + 0] == approx(C(1.0)) );
	CHECK( Arrptr->Zstar_x[0*(Parptr->xsz+1) + 2] == approx(C(4.0)) );
	CHECK( Arrptr->Zstar_x[1*(Parptr->xsz+1) + 0] == approx(C(3.0)) );
	CHECK( Arrptr->Zstar_x[1*(Parptr->xsz+1) + 2] == approx(C(2.0)) );

	CHECK( Arrptr->Zstar_y[0*(Parptr->xsz+1) + 0] == approx(C(1.0)) );
	CHECK( Arrptr->Zstar_y[0*(Parptr->xsz+1) + 1] == approx(C(4.0)) );
	CHECK( Arrptr->Zstar_y[2*(Parptr->xsz+1) + 0] == approx(C(3.0)) );
	CHECK( Arrptr->Zstar_y[2*(Parptr->xsz+1) + 1] == approx(C(2.0)) );
	
	memory_free_legacy(&Arrptr->DEM);
	deallocate_swe_fields(Arrptr);
}

TEST_CASE("Hstar_limits_are_non_negative")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 2;
	Parptr->ysz = 2;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->DEM = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->DEM[0*Parptr->xsz + 0] = C(1.0);
	Arrptr->DEM[0*Parptr->xsz + 1] = C(3.0);
	Arrptr->DEM[1*Parptr->xsz + 0] = C(2.0);
	Arrptr->DEM[1*Parptr->xsz + 1] = C(0.0);

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->H[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->H[0*Parptr->xsz + 1] = C(1.0);
	Arrptr->H[1*Parptr->xsz + 0] = C(1.0);
	Arrptr->H[1*Parptr->xsz + 1] = C(0.0);
	
	allocate_swe_fields(Parptr, Arrptr);
	initialise_Zstar(Parptr, Arrptr);
	update_Hstar(Parptr, Arrptr);

	CHECK( Arrptr->Hstar_neg_x[0*Parptr->xsz + 0] == approx(C(0.0)) );
	CHECK( Arrptr->Hstar_neg_x[0*Parptr->xsz + 1] == approx(C(1.0)) );

	CHECK( Arrptr->Hstar_pos_x[0*Parptr->xsz + 0] == approx(C(0.5)) );
	CHECK( Arrptr->Hstar_pos_x[0*Parptr->xsz + 1] == approx(C(1.0)) );

	CHECK( Arrptr->Hstar_neg_y[0*Parptr->xsz + 0] == approx(C(0.5)) );
	CHECK( Arrptr->Hstar_neg_y[1*Parptr->xsz + 0] == approx(C(1.0)) );

	CHECK( Arrptr->Hstar_pos_y[0*Parptr->xsz + 0] == approx(C(0.0)) );
	CHECK( Arrptr->Hstar_pos_y[1*Parptr->xsz + 0] == approx(C(1.0)) );

	memory_free_legacy(&Arrptr->DEM);
	memory_free_legacy(&Arrptr->H);
	deallocate_swe_fields(Arrptr);
}

TEST_CASE("HUstar_limits_from_Hstar_and_original_velocity")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->DepthThresh = 1e-6;

	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 2;
	Parptr->ysz = 2;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->DEM = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->DEM[0*Parptr->xsz + 0] = C(1.0);
	Arrptr->DEM[0*Parptr->xsz + 1] = C(3.0);
	Arrptr->DEM[1*Parptr->xsz + 0] = C(2.0);
	Arrptr->DEM[1*Parptr->xsz + 1] = C(0.0);

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->H[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->H[0*Parptr->xsz + 1] = C(1.5);
	Arrptr->H[1*Parptr->xsz + 0] = C(1.0);
	Arrptr->H[1*Parptr->xsz + 1] = C(0.0);

	allocate_swe_fields(Parptr, Arrptr);
	Arrptr->HU[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->HU[0*Parptr->xsz + 1] = C(2.0);
	Arrptr->HU[1*Parptr->xsz + 0] = C(1.5);
	Arrptr->HU[1*Parptr->xsz + 1] = C(0.0);

	initialise_Zstar(Parptr, Arrptr);
	update_Hstar(Parptr, Arrptr);

	CHECK( HUstar_neg_x(Parptr, Solverptr, Arrptr, 0, 0) == approx(C(0.0)) );
	CHECK( HUstar_neg_x(Parptr, Solverptr, Arrptr, 1, 0) == approx(C(2.0)) );

	CHECK( HUstar_pos_x(Parptr, Solverptr, Arrptr, 0, 0) == approx(C(0.5)) );
	CHECK( HUstar_pos_x(Parptr, Solverptr, Arrptr, 1, 0) == approx(C(2.0)) );
	
	CHECK( HUstar_neg_y(Parptr, Solverptr, Arrptr, 0, 0) == approx(C(0.5)) );
	CHECK( HUstar_neg_y(Parptr, Solverptr, Arrptr, 0, 1) == approx(C(1.5)) );
	
	CHECK( HUstar_pos_y(Parptr, Solverptr, Arrptr, 0, 0) == approx(C(0.0)) );
	CHECK( HUstar_pos_y(Parptr, Solverptr, Arrptr, 0, 1) == approx(C(1.5)) );

	memory_free_legacy(&Arrptr->DEM);
	memory_free_legacy(&Arrptr->H);
	deallocate_swe_fields(Arrptr);
}

TEST_CASE("Zdagger_limits_ensures_bed_not_higher_than_free_surface_elevation")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 2;
	Parptr->ysz = 2;

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->DEM = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->DEM[0*Parptr->xsz + 0] = C(1.0);
	Arrptr->DEM[0*Parptr->xsz + 1] = C(4.0);
	Arrptr->DEM[1*Parptr->xsz + 0] = C(3.0);
	Arrptr->DEM[1*Parptr->xsz + 1] = C(2.0);

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->H[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->H[0*Parptr->xsz + 1] = C(1.0);
	Arrptr->H[1*Parptr->xsz + 0] = C(1.0);
	Arrptr->H[1*Parptr->xsz + 1] = C(0.0);

	allocate_swe_fields(Parptr, Arrptr);
	initialise_Zstar(Parptr, Arrptr);

	CHECK( Zdagger_neg_x(Parptr, Arrptr, 0, 0) == approx(C(1.5)) );
	CHECK( Zdagger_neg_x(Parptr, Arrptr, 1, 0) == approx(C(4.0)) );

	CHECK( Zdagger_pos_x(Parptr, Arrptr, 0, 0) == approx(C(1.0)) );
	CHECK( Zdagger_pos_x(Parptr, Arrptr, 1, 0) == approx(C(4.0)) );

	CHECK( Zdagger_neg_y(Parptr, Arrptr, 0, 0) == approx(C(1.0)) );
	CHECK( Zdagger_neg_y(Parptr, Arrptr, 0, 1) == approx(C(3.0)) );

	CHECK( Zdagger_pos_y(Parptr, Arrptr, 0, 0) == approx(C(1.5)) );
	CHECK( Zdagger_pos_y(Parptr, Arrptr, 0, 1) == approx(C(3.0)) );

	memory_free_legacy(&Arrptr->DEM);
	memory_free_legacy(&Arrptr->H);
	deallocate_swe_fields(Arrptr);
}

TEST_CASE("bed_source_x")
{
	Pars pars;
	Pars *Parptr = &pars;
	Parptr->xsz = 2;
	Parptr->ysz = 1;
	Parptr->dx = C(4.0);

	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->g = C(9.8);

	Arrays arrays;
	Arrays *Arrptr = &arrays;

	Arrptr->DEM = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->DEM[0*Parptr->xsz + 0] = C(2.0);
	Arrptr->DEM[0*Parptr->xsz + 1] = C(4.0);

	Arrptr->H = memory_allocate_numeric_legacy(Parptr->xsz*Parptr->ysz);
	Arrptr->H[0*Parptr->xsz + 0] = C(1.0);
	Arrptr->H[0*Parptr->xsz + 1] = C(0.0);

	allocate_swe_fields(Parptr, Arrptr);
	Arrptr->Hstar_neg_x[0*Parptr->xsz + 0] = C(0.5);
	Arrptr->Hstar_pos_x[0*Parptr->xsz + 0] = C(1.5);

	initialise_Zstar(Parptr, Arrptr);

	CHECK( bed_source_x(Parptr, Solverptr, Arrptr, 0, 0) == approx(C(-2.45)) );

	memory_free_legacy(&Arrptr->DEM);
	memory_free_legacy(&Arrptr->H);
	deallocate_swe_fields(Arrptr);
}
