#pragma once

#include "catch.hpp"
#include "../lisflood.h"

static NUMERIC_TYPE margin = 1e-9;
static Approx approx = Approx::custom().margin(margin);
