#include "catch.hpp"
#include "checks.h"
#include "../utility.h"
#include "../swe/dg2.h"
#include "../swe/dg2/slope_limiter.h"

using namespace dg2;

TEST_CASE("slope_limiter_zeros_slope_in_discontinuous_zig_zag")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->maxH = C(1.0);
	Solverptr->krivodonova_threshold = C(0.0);

	NUMERIC_TYPE dx = C(1.0);

	NUMERIC_TYPE slope = limit_slope(Solverptr, C(0.5), dx,
			C(2.0), C(2.0)/SQRT(C(3.0)),
			C(3.0), C(-3.0)/SQRT(C(3.0)),
			C(2.5), C(1.0)/SQRT(C(3.0)));

	CHECK( slope == approx(C(0.0)) );
}

TEST_CASE("slope_limiter_leaves_slope_unmodified")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->maxH = C(1.0);

	NUMERIC_TYPE dx = C(1.0);

	NUMERIC_TYPE slope = limit_slope(Solverptr, C(0.5), dx,
			C(2.0), C(1.0)/SQRT(C(3.0)),
			C(5.0), C(1.5)/SQRT(C(3.0)),
			C(7.5), C(0.5)/SQRT(C(3.0)));

	CHECK( slope == approx(C(1.5)/SQRT(C(3.0))) );
}

TEST_CASE("slope_limiter_reduces_slope")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->maxH = C(1.0);

	NUMERIC_TYPE dx = C(1.0);

	NUMERIC_TYPE slope = limit_slope(Solverptr, C(0.5), dx,
			C(2.0), C(1.0)/SQRT(C(3.0)),
			C(5.0), C(4.0)/SQRT(C(3.0)),
			C(7.5), C(0.5)/SQRT(C(3.0)));

	CHECK( slope == approx(C(2.5)/SQRT(C(3.0))) );
}

TEST_CASE("slope_limiter_reduces_negative_slope")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->maxH = C(1.0);

	NUMERIC_TYPE dx = C(1.0);

	NUMERIC_TYPE slope = limit_slope(Solverptr, C(0.5), dx,
			C(7.5), C(-0.5)/SQRT(C(3.0)),
			C(5.0), C(-4.0)/SQRT(C(3.0)),
			C(2.0), C(-1.0)/SQRT(C(3.0)));

	CHECK( slope == approx(C(-2.5)/SQRT(C(3.0))) );
}

TEST_CASE("slope_limiter_skipped_if_min_H_in_stencil_below_threshold")
{
	Solver solver;
	Solver *Solverptr = &solver;
	Solverptr->maxH = C(100.0);

	NUMERIC_TYPE dx = C(1.0);

	NUMERIC_TYPE slope = limit_slope(Solverptr, C(4.9), dx,
			C(2.0), C(1.0)/SQRT(C(3.0)),
			C(5.0), C(4.0)/SQRT(C(3.0)),
			C(7.5), C(0.5)/SQRT(C(3.0)));

	CHECK( slope == approx(C(4.0)/SQRT(C(3.0))) );
}
