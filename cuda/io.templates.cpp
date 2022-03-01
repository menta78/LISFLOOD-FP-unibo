template<typename F>
void lis::AsciiRaster::write
(
	FILE* file,
	F array,
	Geometry& geometry,
	int pitch,
	int offset,
	NUMERIC_TYPE no_data_value,
	int precision
)
{
    fprintf(file, "ncols         %i\n", geometry.xsz);
    fprintf(file, "nrows         %i\n", geometry.ysz);
    fprintf(file, "xllcorner     %.*" NUM_FMT"\n", precision, geometry.blx);
    fprintf(file, "yllcorner     %.*" NUM_FMT"\n", precision, geometry.bly);
    fprintf(file, "cellsize      %.*" NUM_FMT"\n", precision, geometry.dx);
    fprintf(file, "NODATA_value  %.*" NUM_FMT"\n", precision, no_data_value);

	for (int j=0; j<geometry.ysz; j++)
	{
		for (int i=0; i<geometry.xsz; i++)
		{
			fprintf(file, "%.*" NUM_FMT "\t", precision,
					array[j*pitch + i + offset]);
		}
		fprintf(file, "\n");
	}
}

template<typename F>
std::future<void> lis::Snapshot::write_async
(
	F array,
	Geometry& geometry,
	int pitch,
	int offset,
	const char* prefix,
	int counter,
	const char* suffix,
	int verbose,
	int precision,
	NUMERIC_TYPE no_data_value
)
{
	return std::async(std::launch::async, [=, &geometry]{
		Snapshot::write(array, geometry, pitch, offset, prefix, counter, suffix,
				verbose, precision, no_data_value);
	});
}

template<typename F>
void lis::Snapshot::write
(
	F array,
	Geometry& geometry,
	int pitch,
	int offset,
	const char* prefix,
	int counter,
	const char* suffix,
	int verbose,
	int precision,
	NUMERIC_TYPE no_data_value
)
{
	char filename[800];
	snprintf(filename, 800*sizeof(char), "%s-%.4d%s", prefix, counter, suffix);

	FILE* file = fopen_or_die(filename, "wb");
	
	AsciiRaster::write(file, array, geometry, pitch, offset, no_data_value,
			precision);

	fclose(file);
}

