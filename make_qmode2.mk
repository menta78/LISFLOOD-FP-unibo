variants_var += lisflood_float_qmode2

lisflood_float_qmode2: $(MAINOBJS:%.o=lisflood_float_qmode2_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_float_qmode2_obj/%.o) -o lisflood_float_qmode2

$(MAINOBJS:%.o=lisflood_float_qmode2_obj/%.o): lisflood_float_qmode2_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_float_qmode2_obj
	$(MD) -p lisflood_float_qmode2_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -D _NUMERIC_MODE=0 -D _CALCULATE_Q_MODE=2 -c $< -o $@
