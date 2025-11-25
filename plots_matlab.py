# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:35:34 2024

@author: SES271
"""
import matplotlib.pyplot as plt
#%% Set the global font size MATLAB EQUAL
# Use built-in math rendering instead of LaTeX
# plt.style.use(['ggplot', 'grid'])
# Set the font family and font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

# Set the global line width for plots
plt.rcParams['lines.linewidth'] = 1

# Set the global marker size for plots
plt.rcParams['lines.markersize'] = 3
plt.rcParams.update({
    "text.usetex": True
})

# Set the global color cycle for lines
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#4190f0','#DE8F05','#029E73','#D55E00' ])
#['#4190f0','#DE8F05','#029E73','#D55E00' ]
#['#E6E619', '#33CC33', '#3399CC', '#CC3333' ] # For big paper
#['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE']
#['#E6E619', '#33CC33', '#3399CC', '#CC3333' ]
# Set the global line styles (solid, dashed, dotted, etc.)
plt.rcParams['lines.linestyle'] = '-'

# Set the global figure size
plt.rcParams['figure.figsize'] = (8, 6)

# Set the global grid style
# plt.rcParams['grid.linestyle'] = '-'
# plt.rcParams['grid.alpha'] = 0.7

# Set the global legend style
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = '0.8'
plt.rcParams['legend.loc'] = 'best'

# Set the global title position
plt.rcParams['axes.titlepad'] = 20

# Set the global axis label pad
plt.rcParams['axes.labelpad'] = 10

# Set the global x and y axis label styles
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'

# Set the global tick label size
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Set the global legend font size
plt.rcParams['legend.fontsize'] = 16

# Set the global legend frame style
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = 1

# Set the global legend title font size
plt.rcParams['legend.title_fontsize'] = '16'

# Set the global legend spacing
plt.rcParams['legend.labelspacing'] = 0.5

# Set the global legend title spacing
plt.rcParams['legend.handletextpad'] = 0.5

# Set the global grid visibility
plt.rcParams['axes.grid'] = True

# Set the global grid color
#plt.rcParams['grid.color'] = '0.8'


# Set the global axis line width
plt.rcParams['axes.linewidth'] = 0.3

# Set the global x and y axis position
plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05

# Set the global legend column spacing
plt.rcParams['legend.columnspacing'] = 1.0

# Set the global legend handle length
plt.rcParams['legend.handlelength'] = 2.0

# Set the global legend handle height
plt.rcParams['legend.handleheight'] = 0.7

# Set the global legend border pad
plt.rcParams['legend.borderpad'] = 0.4

# Set the global legend border axes pad
plt.rcParams['legend.borderaxespad'] = 0.5

# Set the global legend numpoints
plt.rcParams['legend.numpoints'] = 1

# Set the global legend scatter points
plt.rcParams['legend.scatterpoints'] = 1

# Set the global legend label spacing
plt.rcParams['legend.labelspacing'] = 0.5

# Set the global savefig options
plt.rcParams['savefig.dpi'] = 300

# Set the global savefig format (e.g., 'png', 'pdf', 'svg')
plt.rcParams['savefig.format'] = 'png'

# Set the global savefig bbox_inches
plt.rcParams['savefig.bbox'] = 'tight'

# Set the global savefig pad_inches
plt.rcParams['savefig.pad_inches'] = 0.1

# Set the global savefig transparent
plt.rcParams['savefig.transparent'] = False

# Set the global savefig orientation (landscape, portrait)
plt.rcParams['savefig.orientation'] = 'portrait'