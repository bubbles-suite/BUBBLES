#!/usr/bin/gnuplot

set k bottom

set log x

set xl 'Simulation time [internal units] (Log scale)'
set yl 'Number of adsorbed proteins'

plot 'results/ads/adsorptionLog.dat' u 1:2 w lp t 'Adsorbed proteins'

set t post eps enhanced solid color
set output 'AdsorptionLog.eps'
rep
set output
