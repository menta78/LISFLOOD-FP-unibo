#############################
Bridge tests and directories
#############################
Mark Trigg September 2012
#############################

## hecras ##
hecras model with setups to compare with lisflood

## nswe ##
These are simple tests of nswe flow directions. These were used to develop the uber 4 way model, but can be used on their own to test flow directions with a simple channelx4 in subgrid and 2d dem only channelx4. There are two versions. One that uses 4 separate flow inputs, one to each channel, and another that uses a single flow input into a cell in the centre that each channel shares and then lisflood should divide the flow equally between the channels. The shared input model highlighted a bug in the order of the updateH() code which has now been fixed.

## uber 4 way bridge and weir ##
test of 4 simple channels each starting in the nearly the centre of the domain and flowing out to the edge in N, S, W, E directions. each channel has a bridge and a weir and they are all identical except for the direction of flow. It is a steady state model and the longitudinal plots of final water elevation are compared to each other and hecras in the excel file.

## orifice test ##
Hydrodynamic test of orifice flow. Simple straight subgrid channel with bridge deck. Stage file is plotted to compare with hecras at 6 points along channel in stage_check.xls file. Of most interest is the point just upstream of the bridge. Channel with no bridge is also compared. Finally, there is a staggered steady state run with multiple steady state flows. This allows comparison between hecras and lisflood as well as checks the transition zone results.


