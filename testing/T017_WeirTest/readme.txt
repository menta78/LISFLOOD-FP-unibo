#############################
weir tests and directories
#############################
Mark Trigg September 2012
#############################

## hecras ##
hecras model with setups to compare with lisflood

## simple weir ##
Hydrodynamic test of weir flow. Simple straight subgrid channel with weir crest. Stage file is plotted to compare with hecras at 6 points along channel in weir_stage_check.xls file. Of most interest is the point just upstream of the weir. Channel with no weir is also compared. Note that there seems to be a local stability issue if there is floodplain flow just upstream of the weir and then it is forced back into the channel to go over the weir. We are also not sure that the modulus implementation is totally correct - this should be checked at some point if it is important to your model.

## uber 4 way bridge and weir ##
This is in the bridge testing directory T016, but has a 4 way test of a weir as well as bridge.


