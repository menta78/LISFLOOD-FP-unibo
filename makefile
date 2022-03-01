-include $(CONFIG)

NVCC = nvcc

SRCS := \
boundary.cpp \
ch_flow.cpp \
chkpnt.cpp \
fp_acc.cpp \
fp_flow.cpp \
fp_trent.cpp \
infevap.cpp \
input.cpp \
iterateq.cpp \
output.cpp \
pars.cpp \
por_flow.cpp \
sgc.cpp \
util.cpp \
utility.cpp \
weir_flow.cpp \
lisflood2/DataTypes.cpp \
lisflood2/file_tool.cpp \
lisflood2/lis2_output.cpp \
lisflood2/lisflood_processing.cpp \
lisflood2/sgm_fast.cpp \
lisflood2/windows_ext.cpp \
rain/rain.cpp \
swe/boundary.cpp \
swe/dg2.cpp \
swe/dg2new.cpp \
swe/fields.cpp \
swe/flux.cpp \
swe/fv1.cpp \
swe/hll.cpp \
swe/input.cpp \
swe/output.cpp \
swe/stats.cpp \
swe/dg2/dg2_output.cpp \
swe/dg2/fields.cpp \
swe/dg2/friction.cpp \
swe/dg2/modifiedvars.cpp \
swe/dg2/slope_limiter.cpp \
swe/fv1/modifiedvars.cpp

TEST_SRCS := \
test/test_lisflood.cpp \
test/test_dg2.cpp \
test/test_flux.cpp \
test/test_fv1.cpp \
test/test_slope_limiter.cpp

ifdef CUDA
  NVCCFLAGS += -Icuda/common -Icuda -ccbin $(CXX)
  CXXFLAGS += -Icuda/common -Icuda -DCUDA
  SRCS += \
cuda/cuda_boundary.cu \
cuda/cuda_dem.cu \
cuda/cuda_flow.cu \
cuda/cuda_geometry.cu \
cuda/cuda_hll.cu \
cuda/cuda_max_field.cu \
cuda/cuda_unifiedallocator.cu \
cuda/cuda_rain.cu \
cuda/cuda_sample.cu \
cuda/cuda_simulate.cu \
cuda/cuda_solver.cu \
cuda/cuda_snapshot.cu \
cuda/cuda_stats.cu \
cuda/cuda_util.cu \
cuda/ghostraster.cpp \
cuda/io.cpp \
cuda/sample.cpp \
cuda/stats.cpp \
cuda/fv1/cuda_fv1_flow.cu \
cuda/fv1/cuda_fv1_simulate.cu \
cuda/fv1/cuda_fv1_snapshot.cu \
cuda/fv1/cuda_fv1_solver.cu \
cuda/dg2/cuda_dg2_dem.cu \
cuda/dg2/cuda_dg2_flow.cu \
cuda/dg2/cuda_dg2_simulate.cu \
cuda/dg2/cuda_dg2_slope_limit.cu \
cuda/dg2/cuda_dg2_snapshot.cu \
cuda/dg2/cuda_dg2_solver.cu
endif

OBJDIR := build
DEPDIR := dep

define objs
$(patsubst %,$(OBJDIR)/%.o,$(basename $(1)))
endef

define deps
$(patsubst %,$(DEPDIR)/%.d,$(basename $(1)))
endef

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

COMPILE.cxx = $(CXX) $(DEPFLAGS) $(CXXFLAGS) -c $< -o $@
COMPILE.cu = $(NVCC) $(NVCCFLAGS) --compiler-options "$(CXXFLAGS)" -dc $< -o $@
GENDEPS.cu = $(CXX) -x c++ -M $(DEPFLAGS) -I$(CUDA_HOME)/include $(CXXFLAGS) -c $<

ifdef CUDA
  LINK.o = $(NVCC) $(NVCCFLAGS) --linker-options "$(LDFLAGS)" \
	  $$^ -o $$@ $(LDLIBS)
else
  LINK.o = $(CXX) $(LDFLAGS) $$^ -o $$@ $(LDLIBS)
endif

ALLSRCS += $(SRCS)

# $(1) -- executable
# $(2) -- source files
define app
ALLSRCS += $(2)

$(1): $(call objs,$(2) $(SRCS))
	$(LINK.o)
endef

$(eval $(call app,lisflood,lisflood.cpp))
$(eval $(call app,test_lisflood,$(TEST_SRCS)))
$(eval $(call app,create1DDamBreak,preprocess/create1DDamBreak.cpp))
$(eval $(call app,createDamBreakObstacle,preprocess/createDamBreakObstacle.cpp))
$(eval $(call app,createHump,preprocess/createHump.cpp))
$(eval $(call app,createLakeAtRestCones,preprocess/createLakeAtRestCones.cpp))
$(eval $(call app,createLakeAtRestBlocks,preprocess/createLakeAtRestBlocks.cpp))
$(eval $(call app,createRadialDamBreak,preprocess/createRadialDamBreak.cpp))
$(eval $(call app,checkDynamicRain,preprocess/checkDynamicRain.cpp))
$(eval $(call app,generateDG2DEM,preprocess/generateDG2DEM.cpp))
$(eval $(call app,hll_util,preprocess/hll_util.cpp))

ALLOBJS := $(call objs,$(ALLSRCS))
ALLDEPS := $(call deps,$(ALLSRCS))

$(shell mkdir -p $(dir $(ALLOBJS)) >/dev/null)
$(shell mkdir -p $(dir $(ALLDEPS)) >/dev/null)

$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	$(COMPILE.cxx)

$(OBJDIR)/%.o: %.cu
$(OBJDIR)/%.o: %.cu $(DEPDIR)/%.d
	$(GENDEPS.cu)
	$(COMPILE.cu)

.PHONY: test
test: test_lisflood
	./$<

.PHONY: clean
clean:
	$(RM) -r $(OBJDIR) $(DEPDIR)

.PRECIOUS: $(DEPDIR)/%.d
$(DEPDIR)/%.d: ;

-include $(ALLDEPS)
