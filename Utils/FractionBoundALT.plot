set title 'Transferrin on PS'

set yl 'Fraction Bound'
set xl '[Tf]/[NP]'


set log 

set xr[1:2e4]
set yr[0.5:1.3]

set xtics 10
set mxtics 10
set ytics 10
set mytics 10

set ytics add ("0.9" 0.9, "0.8" 0.8, "0.7" 0.7, "0.6" 0.6, "0.5" 0.5, "0.4" 0.4, "0.3" 0.3)

  plot './FractionBoundALT.dat' u 1:($2/$1) ps 1.5 pt 1 w p t 'Only HC'
replot './FractionBoundALT.dat' u 1:(($2 + $3)/$1) ps 1.5 pt 2 w p t 'HC + SC 1 layer'
replot './FractionBoundALT.dat' u 1:(($2 + $3 + $4)/$1) ps 1.5 pt 4 w p t 'HC + SC 2 layers'

set style arrow 1 nohead lt 2
set style arrow 2 nohead lt 1
set style arrow 3 nohead lt 1 lw 0.5
set style arrow 4 nohead lt 1 lw 2.5

set arrow from 1,1 to 1e4,1 as 1

set arrow from 380,0.5 to 380,1.02 as 1
#set arrow from 400,1 to 8e3,0.5 as 4
#set arrow from 400,1 to 800,0.5 as 3

rep x>400 ? 380/x : 0/0 w l lt 1 lw 0.5 t 'Strong Binding Model'
rep x>400 ? (380/x)**0.25 : 0/0 w l lt 1 lw 2.5 t 'Soft Binding Model'

set size ratio 0.8

set t post eps enhanced "Serif" 20

set output 'FractionBoundALT.eps'
rep
set output

set t wxt

rep
