variants_var += lisflood_float_qmode1

lisflood_float_qmode1: $(MAINOBJS:%.o=lisflood_float_qmode1_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_float_qmode1_obj/%.o) -o lisflood_float_qmode1

$(MAINOBJS:%.o=lisflood_float_qmode1_obj/%.o): lisflood_float_qmode1_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_float_qmode1_obj
	$(MD) -p lisflood_float_qmode1_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -D _NUMERIC_MODE=0 -D _CALCULATE_Q_MODE=1 -c $< -o $@
