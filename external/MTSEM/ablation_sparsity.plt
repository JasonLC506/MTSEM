reset
set grid
set size 0.9,0.9
set datafile separator "\t"
set terminal epslatex color standalone font ",12"
set key right top
# set style fill transparent solid 0.7 border -1
set yrang[1.5:2.8]
set xrang [1e-10:2e-1]
set xtics ("0" 1e-10, "1e-5" 1e-5, "1e-4" 1e-4, "1e-3" 1e-3, "1e-2" 1e-2, "1e-1" 1e-1)
set ytics ("1.5" 1.5, "2.0" 2.0, "2.5" 2.5)
set logscale x
set xlabel '$\lambda$'
set ylabel 'miss-classification rate'
set style line 1 lt 1 lc rgb "#e41a1c" lw 9 pt 3 ps 4 
set style line 2 lt 1 lc rgb "#377eb8" lw 9 pt 6 ps 4 
set style line 3 lt 1 lc rgb "#4daf4a" lw 9 pt 8 ps 4 
set style line 4 lt 1 lc rgb "#984ea3" lw 9 pt 12 ps 4 
set style line 5 lt 1 lc rgb "#DD6500" lw 9 pt 4 ps 4
set style line 7 lt 1 lc rgb "#a65628" lw 9 pt 14 ps 4 
Shadecolor1 = "#e41a1c"
Shadecolor2 = "#377eb8"
Shadecolor3 = "#4daf4a"
Shadecolor4 = "#984ea3"
set output 'ablationsparsity.tex'
plot 'ablation_sparsity.txt' using 1:2 with linespoints linestyle 1 title "TMTS-el", \
'' using 1:3 with linespoints linestyle 2 title "TMTS-ex"

unset output  
set output
system('latex ablationsparsity.tex && dvips ablationsparsity.dvi && ps2pdf ablationsparsity.ps && mv ablationsparsity.ps ablationsparsity.eps')
unset terminal
reset
