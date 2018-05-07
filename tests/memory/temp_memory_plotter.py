
from .context import pibronic
import matplotlib as mpl
#mpl.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import sys, os



memory_data = np.loadtxt("mprofile_20161117134114.dat", skiprows=1, usecols=(1,2))
# the initial time 
time_value = memory_data[0,1]

# convert to GB
memory_data[:, 0] *= 0.00104858

figure, ax = plt.subplots(1, 1)

ax.plot(memory_data[:, 1], memory_data[:, 0],  color='b',)
#ax.axhline(y=true_value[temp_index], linewidth=2, color='r', label='Analytically derived ratio')

time_value += 8.870823
ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='r', label='Start of numerator')
time_value += 1.1968180000000004
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='g', label='Building O matrix')
time_value += 0.06710700000000003
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='y', label='Modifying q tensor')
time_value += 22.284717
ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='r', label='Making coupling matrix')
time_value += 0.0003700000000037562
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='g', label='Allocating eigvals and eigvects')
time_value += 98.80051099999999
ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='y', label='Diagonalizing coupling matrix')
time_value += 0.1403500000000122
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='r', label='Constructing M matrix')
time_value += 0.00042199999998615567
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='g', label='Allocating temporary numerator tensor')
time_value += 56.78094800000002
ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='y', label='Multiplying O and M matricies together')
time_value += 0.00042799999999942884
#ax.axvline(x=time_value, linewidth=1, alpha=0.8, color='r', label='Taking the trace')

ax.set_ylabel("Memory used" , fontsize=14)
ax.set_xlabel("time (seconds)")

#ax.set_xscale('log') # doesn't work when data is negative
#ax.set_xlim(int(1e-2), int(1e2))
#ax.legend(loc='upper left', fontsize=10)

# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()

# Remove the plot frame lines. They are unnecessary chartjunk. 
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  

#ax.set_ylim(-8., 8.)

figure.set_size_inches(14, 8)
figure.savefig("memory_profile.pdf".format(), transparent=True, bbox_inch='tight', pad_inches=0)