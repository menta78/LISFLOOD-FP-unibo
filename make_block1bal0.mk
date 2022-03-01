variants_var += lisflood_float_blk1_bal0

lisflood_float_blk1_bal0: $(MAINOBJS:%.o=lisflood_float_blk1_bal0_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_float_blk1_bal0_obj/%.o) -o lisflood_float_blk1_bal0

$(MAINOBJS:%.o=lisflood_float_blk1_bal0_obj/%.o): lisflood_float_blk1_bal0_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_float_blk1_bal0_obj
	$(MD) -p lisflood_float_blk1_bal0_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -D _NUMERIC_MODE=0 -D _SGM_BY_BLOCKS=1 -D _BALANCE_TYPE=0 -c $< -o $@
