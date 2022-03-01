variants_var += lisflood_float_disable_vec

# Linkage rule FLOAT NO VECT
lisflood_float_disable_vec: $(MAINOBJS:%.o=lisflood_float_disable_vec_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_float_disable_vec_obj/%.o)  -o lisflood_float_disable_vec

# Compile main object files FLOAT NO VECT
$(MAINOBJS:%.o=lisflood_float_disable_vec_obj/%.o): lisflood_float_disable_vec_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_float_disable_vec_obj
	$(MD) -p lisflood_float_disable_vec_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -no-vec -no-simd -D _NUMERIC_MODE=0 -c $< -o $@

