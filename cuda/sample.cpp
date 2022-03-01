#include "sample.h"
#include <algorithm>

void lis::Sample::initialise
(
	SamplePoints& sample_points,
	const char* filename,
	Geometry& geometry,
	int pitch,
	int offset,
	int verbose
)
{
	FILE* file = fopen_or_die(filename, "rb", "Loading stagefile\n", verbose);

	fscanf(file, "%d", &(sample_points.count));
	fgetc(file); // skip EOL

	sample_points.x = new NUMERIC_TYPE[sample_points.count]();
	sample_points.y = new NUMERIC_TYPE[sample_points.count]();
	sample_points.idx = new int[sample_points.count]();
	sample_points.inside_domain = new bool[sample_points.count]();

	for (int p=0; p<sample_points.count; p++)
	{
		NUMERIC_TYPE& x = sample_points.x[p];
		NUMERIC_TYPE& y = sample_points.y[p];

		fscanf(file, "%" NUM_FMT, &x);
		fscanf(file, "%" NUM_FMT, &y);

		sample_points.inside_domain[p] = (x >= geometry.blx &&
				x <= geometry.blx + geometry.xsz*geometry.dx &&
				y >= geometry.blx && y <= geometry.tly);

		int i = static_cast<int>(floor((x - geometry.blx) / geometry.dx));
		int j = geometry.ysz - 1 - static_cast<int>(
				floor((y - geometry.bly) / geometry.dy));

		i = std::min(i, geometry.xsz-1);
		j = std::max(j, 0);

		sample_points.idx[p] = j*pitch + i + offset;
	}

	fclose(file);
}

void lis::Sample::free
(
	SamplePoints& sample_points
)
{
	delete[] sample_points.x;
	delete[] sample_points.y;
	delete[] sample_points.idx;
	delete[] sample_points.inside_domain;
}

lis::StageFile::StageFile
(
	SamplePoints& sample_points
)
:
sample_points(sample_points)
{}

void lis::StageFile::open
(
	const char* resroot
)
{
	char filename[800];
	snprintf(filename, 800*sizeof(char), "%s%s", resroot, ".stage");
	file = fopen_or_die(filename, "wb");
}

void lis::StageFile::write_header
(
	NUMERIC_TYPE* DEM,
	const char* sample_points_filename
)
{
	fprintf(file, "Stage output, depth (m). Stage locations from: %s\n\n",
			sample_points_filename);
	fprintf(file, "Stage information (stage,x,y,elev):\n");

	for (int i=0; i<sample_points.count; i++)
	{
		if (sample_points.inside_domain[i])
		{	
			fprintf(file,
					"%d\t%.4" NUM_FMT"\t%.4" NUM_FMT"\t%.4" NUM_FMT"\n", i+1,
					sample_points.x[i], sample_points.y[i],
					DEM[sample_points.idx[i]]);
		}
		else
		{
			fprintf(file, "%d\t%.4" NUM_FMT"\t%.4" NUM_FMT"\tn/a\n", i+1,
					sample_points.x[i], sample_points.y[i]);
		}
	}
	fprintf(file, "\nOutput, depths:\n");
	fprintf(file, "Time; stages 1 to %d\n", sample_points.count);
}

void lis::StageFile::write
(
	SampleBuffer& sample_buf,
	int sample_buf_idx
)
{
	for (int i=0; i<sample_buf_idx; i++)
	{
		fprintf(file, "%12.3" NUM_FMT "\t", sample_buf.time[i]);

		int offset = i * sample_points.count;

		for (int p=0; p < sample_points.count; p++)
		{
			if (sample_points.inside_domain[p])
			{
				fprintf(file, "%10.4" NUM_FMT "\t",
						sample_buf.H[offset+p]);
			}
			else
			{
				fprintf(file, "-\t");
			}
		}
		fprintf(file, "\n");
	}
	fflush(file);
}

lis::StageFile::~StageFile()
{
	if (file != nullptr) fclose(file);
}

lis::GaugeFile::GaugeFile
(
	SamplePoints& sample_points
)
:
sample_points(sample_points)
{}

void lis::GaugeFile::open
(
	const char* resroot
)
{
	char filename[800];
	snprintf(filename, 800*sizeof(char), "%s%s", resroot, ".velocity");
	file = fopen_or_die(filename, "wb");
}

void lis::GaugeFile::write_header
(
	NUMERIC_TYPE* DEM,
	const char* sample_points_filename
)
{
	fprintf(file, "Velocity output, velocity (ms-1). Velocity locations from: %s\n\n", sample_points_filename);
	fprintf(file, "Stage information (stage,x,y,elev):\n");

	for (int i=0; i<sample_points.count; i++)
	{
		if (sample_points.inside_domain[i])
		{	
			fprintf(file,
					"%d\t%.4" NUM_FMT"\t%.4" NUM_FMT"\t%.4" NUM_FMT"\n", i+1,
					sample_points.x[i], sample_points.y[i],
					DEM[sample_points.idx[i]]);
		}
		else
		{
			fprintf(file, "%d\t%.4" NUM_FMT"\t%.4" NUM_FMT"\tn/a\n", i+1,
					sample_points.x[i], sample_points.y[i]);
		}
	}

	fprintf(file, "\nOutput, depths:\n");
	fprintf(file, "Time; velocities 1 to %d\n", sample_points.count);
}

void lis::GaugeFile::write
(
	SampleBuffer& sample_buf,
	int sample_buf_idx
)
{
	for (int i=0; i<sample_buf_idx; i++)
	{
		fprintf(file, "%12.3" NUM_FMT "\t", sample_buf.time[i]);

		int offset = i * sample_points.count;

		for (int p=0; p < sample_points.count; p++)
		{
			if (sample_points.inside_domain[p])
			{
				fprintf(file, "%10.4" NUM_FMT "\t",
						sample_buf.speed[offset+p]);
			}
			else
			{
				fprintf(file, "-\t");
			}
		}
		fprintf(file, "\n");
	}
	fflush(file);
}

lis::GaugeFile::~GaugeFile()
{
	if (file != nullptr) fclose(file);
}
