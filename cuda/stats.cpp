#include "stats.h"
#include "../lisflood.h"

lis::Stats::Stats
(
	const char* resroot,
	NUMERIC_TYPE interval,
	NUMERIC_TYPE next_save
)
:
interval(interval),
next_save(next_save)
{
	char filename[800];
	snprintf(filename, 800*sizeof(char), "%s%s", resroot, ".mass");
	file = fopen_or_die(filename, "w");
}

void lis::Stats::write_header()
{
	fprintf(file, "Time         Tstep      MinTstep   NumTsteps    Area         Vol         Qin         Hds        Qout          Qerror       Verror       Rain-(Inf+Evap)\n");
}

bool lis::Stats::need_to_write
(
	NUMERIC_TYPE t
)
{
	return t >= next_save;
}

void lis::Stats::write
(
	StatsEntry entry
)
{
	fprintf
	(
		file,
		"%-12.3f %10.4e %10.4e %-10li %12.4e %12.4e  %-11.3" NUM_FMT" %-10.3" NUM_FMT" %-11.3" NUM_FMT" %12.4e %12.4e %12.4e\n",
		entry.t,
		entry.dt,
		entry.min_dt,
		entry.iteration,
		entry.area,
		entry.volume,
		entry.mass.in,
		C(0.0) /*Solverptr->Hds*/,
		entry.mass.out,
		entry.discharge_error,
		entry.volume_error,
		C(0.0) /*Parptr->RainTotalLoss - (Parptr->InfilTotalLoss + Parptr->EvapTotalLoss)*/
	);
	fflush(file);

	next_save += interval;
}

lis::Stats::~Stats()
{
	fclose(file);
}
