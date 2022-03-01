BUILD_DIR=~/netcdf_build_intel

# http://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html
# http://www.hdfgroup.org/HDF5/release/obtainsrc.html
# cd ~/${BUILD_DIR}
# wget http://zlib.net/zlib-1.2.8.tar.gz
# wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.14.tar.gz
# wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.3.2.tar.gz

#tar zxvf zlib-1.2.8.tar.gz
#tar zxvf hdf5-1.8.14.tar.gz
#tar zxvf netcdf-4.3.2.tar.gz

RELEASE_DIR=${BUILD_DIR}/release

ZLIB_SOURCE=${BUILD_DIR}/zlib-1.2.8
HDF5_SOURCE=${BUILD_DIR}/hdf5-1.8.14
NETCDF_SOURCE=${BUILD_DIR}/netcdf-4.3.2

mkdir ${RELEASE_DIR}

export CC=icc
export CFLAGS='-O3 -xHost -ip'
export CXX=icpc
export CXXFLAGS='-O3 -xHost -ip'

#cd ${ZLIB_SOURCE}
#./configure --prefix=${RELEASE_DIR}
#make check install


cd ${HDF5_SOURCE}
./configure --with-zlib=${RELEASE_DIR} --prefix=${RELEASE_DIR}
#make check install
#make install


cd ${NETCDF_SOURCE}
CPPFLAGS="-I${RELEASE_DIR}/include" \
LDFLAGS="-L${RELEASE_DIR}/lib" \
./configure --prefix=${RELEASE_DIR}
make check install
