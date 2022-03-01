variants_var += lisflood_float_disable_wet_dry

lisflood_float_disable_wet_dry: $(MAINOBJS:%.o=lisflood_float_disable_wet_dry_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_float_disable_wet_dry_obj/%.o) -o lisflood_float_disable_wet_dry

$(MAINOBJS:%.o=lisflood_float_disable_wet_dry_obj/%.o): lisflood_float_disable_wet_dry_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_float_disable_wet_dry_obj
	$(MD) -p lisflood_float_disable_wet_dry_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -D _NUMERIC_MODE=0 -D _DISABLE_WET_DRY=1 -c $< -o $@
