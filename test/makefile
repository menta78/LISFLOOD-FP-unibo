EXE = test_lisflood

OBJS = test_lisflood.o \
       test_dg2.o \
       test_flux.o \
       test_fv1.o \
       test_slope_limiter.o

.PHONY: clean

# Linkage rule
$(EXE): $(OBJS:%.o=$(EXE)_obj/%.o) $(LISFLOOD_OBJS:%.o=../$(EXE_DOUBLE_GCC)_obj/%.o)
	$(LD_GCC) $(LDFLAGS_GCC) $^ -o $(EXE) $(LDLIBS_GCC)

# Compile main object files
$(OBJS:%.o=$(EXE)_obj/%.o): $(EXE)_obj/%.o: %.cpp
	$(MD) -p $(EXE)_obj
	$(CC_GCC) $(CFLAGS_GCC) -D _NUMERIC_MODE=1 -c $< -o $@

clean:
	$(RM) -rf $(EXE)_obj
