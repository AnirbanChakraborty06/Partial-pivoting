# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:53:21 2021

@author: Chako
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# Location where plots are to be stored and filenames.
PLOT_LOC = r'E:\Numpy_Pandas\Jupyter_ipnyb_files\Partial_Pivoting_Project\Plots'
TABLE_LOC = r'E:\Numpy_Pandas\Jupyter_ipnyb_files\Partial_Pivoting_Project\SimulationData'
FIG_SPIKE_EXISTENCE = '\\SpikeExistence_Size%d'
FIG_SPIKE_POSITION = '\\SpikePosition_Size%d'
TABLE_NAME = r'\simulation_table.csv'


def single_matrix_single_trial(n, pos, low, high, step, stop, statistic):
    '''
    

    Parameters
    ----------
    n : number
        Size of the square matrix.
    pos : tuple
        Position to be considered for study.
    low : number
        Lower limit of matrix entries.
    high : number
        Higher limit of matrix entries.
    step : float
        Change in relative value of entry in each step.
    stop : float
        Lower limit of relative value of entry.
    statistic : function
        A function to be used to derive a measure of the
        entries below the one to be studies.

    Returns
    -------
    list 
    with following three entries - 
        x_data - pd.Series of relative values of entry
        y_data - pd.Series of relative changes of solution
        max_spike_point - relative entry value where max
        spike or instability is observed.

    '''
    
    # Block for pivot position.
    if pos[0] == pos[1] == 0:
        
        # The matrix A generated may turn out to be
        # singular on account of (say) column 2 and 3
        # being dependant. Then any calculations further
        # down the line with A will fail. We want to avoid that.
        while True:
            A = np.array(np.random.randint(low=low, high=high, size=(n,n)), dtype='float64')
            if np.linalg.matrix_rank(A)==n:
                break
        A[pos[0], pos[1]] = high
    
    # Block for non-pivot position
    elif pos[0]==1 and pos[1]==0:
        
        # Similar logic as in the pivot case to avoid random 
        # matrix A being singular.
        
        # Besides, there is one more step involved here. We want
        # the non-pivot position to be studied only after the
        # partial pivoting has been done. Hence we exchange the
        # row with max value in column-1 with the first row.
        while True:
            A = np.array(np.random.randint(low=low, high=high, size=(n,n)), dtype='float64')
            max_row = np.argmax(A[:,pos[1]])
            A[0], A[max_row] = np.copy(A[max_row]), np.copy(A[0])
            if np.linalg.matrix_rank(A)==n:
                break
    
    else:
        raise ValueError('Method not suited to study positions other than (0,0) and (1,0).')
        
    
    # Setting b to be used in Ax=b
    b=np.array(np.random.randint(low=low, high=n*high, size=(n,)), dtype='float64').transpose()
    
    # Finding the measure of the non-pivot entries of
    # pivot column
    col_statistic = statistic(A[pos[0]:,pos[1]])
    
    # Generating the sequence of relative pivot values
    frac_0 = A[pos[0], pos[1]]/col_statistic
    frac_ser = pd.Series(np.arange(start=frac_0, stop = stop, step=-(step)))
    
    # Generating the sequence of actual pivot values
    pivot_ser = frac_ser*col_statistic
    pivot_ser[0] = A[pos[0], pos[1]]
    
    # Generating the sequence of matrices
    matrix_ser = []
    for new_pivot in pivot_ser:
        A_new = np.copy(A)
        A_new[pos[0], pos[1]] = new_pivot
        matrix_ser.append(A_new)
        
    # Generating the sequence of solutions
    soln_ser = pd.DataFrame()
    for i in range(len(matrix_ser)):
        if np.linalg.matrix_rank(matrix_ser[i]) < n:
            frac_ser.drop(labels=[i], inplace=True)
            # print('Deleted index', i, 'with value', frac_ser[i], 'from frac_ser')
        else:
            soln = np.linalg.solve(a=matrix_ser[i], b=b)
            soln_ser.loc[:,i] = soln
    
    # Reindexing frac_ser following possible dropping in above loop
    frac_ser.index = np.arange(len(frac_ser))
    soln_ser.columns = np.arange(len(soln_ser.columns))
    
    # Generating sequence of solution changes
    change_ser = pd.DataFrame()
    for i in range(len(soln_ser.columns)-1):
        change_ser.loc[:,i] = soln_ser[i+1]-soln_ser[i]
        
    # Generating sequence of relative changes in solution
    change_mag_ser = np.sqrt(np.square(change_ser).sum(axis=0))
    soln_mag_ser = np.sqrt(np.square(soln_ser).sum(axis=0))
    
    fractional_chng_ser = pd.Series(dtype='float64')
    
    for i in range(len(change_mag_ser)):
        fractional_chng_ser.loc[i] = (change_mag_ser[i]/soln_mag_ser[i])*(step/(frac_ser[i]-frac_ser[i+1]))

    # Preparing the return values
    x_data = frac_ser[:len(frac_ser)-1]
    y_data = fractional_chng_ser
    percentile_marks = y_data.quantile(q=[0.9, 1.0])
    y_data_90, y_data_100 = percentile_marks[0.9], percentile_marks[1.0]
    max_spike_point = x_data[y_data.idxmax()]
    return [x_data, y_data, max_spike_point, (y_data_90, y_data_100)]

def single_matrix_single_trial_plot(n, pos, low, high, step, stop, statistic):
    '''
    Generate plot of relative solution change values vs
    relative entry values for a single square matrix of
    size n.
    
    The plot is saved at location PLOT_LOC. File name
    pattern is "Single_Matrix_<n>".
    '''
    
    filename = PLOT_LOC + '\\' + f'Size{n}_Pos_{pos[0]}_{pos[1]}_Relative_Val_vs_Instability'
    x_data, y_data, max_spike_point, *other_data = single_matrix_single_trial(n, pos, low, high, step, stop, statistic)
    fig = plt.figure()
    sub1 = fig.add_subplot(1,1,1)
    range_ser = y_data.quantile(q=[0.025, 0.975, 1.0])
    ymin = np.floor(range_ser[0.025])
    ymax = np.ceil(range_ser[0.975])
    plot_props = {
                'ylim': (ymin, ymax),
                'title': f'Dependence of solution stability for entry ({pos[0]},{pos[1]})',
                'xlabel': 'relative entry value',
                'ylabel': 'relative solution change'
    }
    sub1.set(**plot_props)
    sub1.plot(x_data, y_data, color='b', marker='.', linestyle='-')
    fig.savefig(fname=filename)
    



def simulation_data_generation(size_range=(2,10), positions=((0,0),(1,0)), high=10, low=1, step=0.01, stop=0.001, statistic=np.min, num_of_trials=100):
    '''
    

    Parameters
    ----------
    size_range : tuple, optional
        The range of matrix sizes under observation. The default is (2,10).
    positions : tuple, optional
        The tuple of positions under observation. The default is ((0,0),(1,0)).
    high : number, optional
        The maximum matrix entry. The default is 10.
    low : number, optional
        The minimum matrix entry. The default is 1.
    step : float, optional
        The steps in the relative entry values. The default is 0.01.
    stop : float, optional
        The lower limit of relative entry values. The default is 0.001.
    statistic : function, optional
        A function that produces a statistic to 
        measure the entries of a column. The default is np.min.
    num_of_trials : number, optional
        Number of trials per size and position
        in the simulation. The default is 100.

    Returns
    -------
    pd.DataFrame that produces that captures the data
    generated from the simulation.

    '''
    
    # Size and position indexing along the rows
    array_of_sizes = np.arange(size_range[0], size_range[1]+1)
    size_pos_index = pd.MultiIndex.from_product(iterables=[array_of_sizes,positions], names=['size','position'])
    
    # No. of trials and observed properties along the columns
    array_of_trials = np.arange(1,num_of_trials+1)
    obs_property = ['max_spike_point', '90_percentile', '100_percentile']
    data_index = pd.MultiIndex.from_product(iterables=[array_of_trials,obs_property], names=['trials','property'])
    
    # Summary table
    simulation_table = pd.DataFrame(index=size_pos_index, columns=data_index)
    
    # Populating the simulation_table
    for size_pos in simulation_table.index:
        n=size_pos[0]
        pos = size_pos[1]
        start_time = time.time()
        for i in range(1,num_of_trials+1):
            x_data, y_data, max_spike_point, (y_data_90, y_data_100) = single_matrix_single_trial(n, pos, low, high, step, stop, statistic)
            simulation_table.loc[size_pos,(i,'max_spike_point')] = max_spike_point
            simulation_table.loc[size_pos,(i,'90_percentile')] = y_data_90
            simulation_table.loc[size_pos,(i,'100_percentile')] = y_data_100
        end_time = time.time()
        time_gap = round(end_time-start_time,2)
        print(f'size: {n}  pos: {pos}  time: {time_gap}')
    
    # Storing the simulation data in .csv file
    filename = TABLE_LOC + TABLE_NAME
    simulation_table.to_csv(filename)
    # return simulation_table
    
def simulation_plot_gen(data_gen_reqd):
    """
    Generates the plots for the simulated data.

    Parameters
    ----------
    data_gen_reqd : boolean
        Determines if data generation is required or not.

    """
    
    if data_gen_reqd:
        simulation_data_generation()
    filename = TABLE_LOC + TABLE_NAME
    simulation_data = pd.read_csv(filename, header=[0,1], index_col=[0,1])
        
    # Capturing the unique level values of the row and column index
    # from the simulation data. The row-indices are the size and 
    # position (pivot or non-pivot). The column indices are the
    # number of trials and observed properties for each trial like
    # 90-percentile, 100-percentile & max-spike-position.
    pivot_pos = simulation_data.index.unique(level='position')[0]
    non_pivot_pos = simulation_data.index.unique(level='position')[1]
    max_spike_point = simulation_data.columns.unique(level='property')[0]
    percentile_90 = simulation_data.columns.unique(level='property')[1]
    percentile_100 = simulation_data.columns.unique(level='property')[2]
    
    for i,size in enumerate(simulation_data.index.unique(level='size')):
    
        # Capturing the pivot and non-pivot data for each size
        pivot_data = simulation_data.loc[(size, pivot_pos)]
        non_pivot_data = simulation_data.loc[(size,non_pivot_pos)]
        
        # Capturing the properties 90_percentile, 100_percentile
        # and max-spike-position for pivot and non-pivot data.
        pivot_data_90 = pivot_data.loc[:, percentile_90]
        pivot_data_100 = pivot_data.loc[:, percentile_100]
        pivot_max_spike_point = pivot_data.loc[:, max_spike_point]
        
        
        non_pivot_data_90 = non_pivot_data.loc[:, percentile_90]
        non_pivot_data_100 = non_pivot_data.loc[:, percentile_100]
        non_pivot_max_spike_point = non_pivot_data.loc[:, max_spike_point]
        
        # Plotting the pivot_vs_nonpivot_spikeExistence based on percentile data
        pivot_vs_nonpivot_spikeExistence = plt.figure(figsize=[10,4])
        
        pivot_plot = pivot_vs_nonpivot_spikeExistence.add_subplot(1,2,1)
        nonpivot_plot = pivot_vs_nonpivot_spikeExistence.add_subplot(1,2,2)
        
        # Removing the ouliers from below box-plots
        # so that the separation between spread of 90-percentile
        # and 100-percentile marks are evident;
        # otherwise a few outliers hide the actual spread of these
        # data points
        pivot_plot.boxplot([pivot_data_90, pivot_data_100], labels=['90-percentile', '100-percentile'], showfliers=False)
        nonpivot_plot.boxplot([non_pivot_data_90, non_pivot_data_100], labels=['90-percentile', '100-percentile'], showfliers=False)
        pivot_plot.set_title(f'Size={size}   Pivot')
        nonpivot_plot.set_title(f'Size={size}   Non-pivot')
        filename = (PLOT_LOC + FIG_SPIKE_EXISTENCE) % size
        pivot_vs_nonpivot_spikeExistence.savefig(filename)
        plt.close(pivot_vs_nonpivot_spikeExistence)
        
        # Plotting the pivot_spikePosition
        pivot_spikePosition = plt.figure(figsize=[5,4])
        
        spikePosition_plot = pivot_spikePosition.add_subplot(1,1,1)
        
        spikePosition_plot.boxplot([pivot_max_spike_point], labels=['Pivot spike-position'])
        spikePosition_plot.set_title(f'Size={size}')
        filename = (PLOT_LOC + FIG_SPIKE_POSITION) % size
        pivot_spikePosition.savefig(filename)
        plt.close(pivot_spikePosition)
    
    
    
if __name__ == '__main__':
    
    while True:
        data_gen_reqd = input('Is simulation data generation reqd Y/N:    ')
        if data_gen_reqd in ('Y','N'):
            break
    data_gen_reqd = {'Y': True, 'N': False}[data_gen_reqd]
    simulation_plot_gen(data_gen_reqd)

    
    

