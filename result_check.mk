lisflood_double_result_check: $(MAINOBJS:%.o=lisflood_double_result_check_obj/%.o)
	$(LD) $(LDFLAGS) $(MAINOBJS:%.o=lisflood_double_result_check_obj/%.o) -o lisflood_double_result_check

$(MAINOBJS:%.o=lisflood_double_result_check_obj/%.o): lisflood_double_result_check_obj/%.o: %.cpp *.h
	$(MD) -p lisflood_double_result_check_obj
	$(MD) -p lisflood_double_result_check_obj/lisflood2
	$(CC) $(CFLAGS_RELEASE) -D _NUMERIC_MODE=1 -D RESULT_CHECK=1 -c $< -o $@
