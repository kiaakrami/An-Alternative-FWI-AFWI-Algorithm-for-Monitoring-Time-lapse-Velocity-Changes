####### We start by importing some useful packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import sys, getopt
from pysit import *
from pysit.gallery import marmousi
from pysit.util.parallel import *
from mpi4py import MPI
from scipy.io import savemat, loadmat

####### Here we define the plots


def plot_func(fig_nr, arr_2d, x_min, x_max, z_min, z_max, x_label, z_label, title, cbar_min=None, cbar_max=None):
    fig = plt.figure(fig_nr)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr_2d, extent=[x_min,x_max,z_max,z_min], interpolation="nearest")
    im.axes.yaxis.set_label_text(z_label, fontsize = 10)
    im.axes.xaxis.set_label_text(x_label, fontsize = 10)
    im.axes.set_title(title, fontsize = 10)

    if cbar_min !=None and cbar_max !=None:
        norm =  mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
        im.set_norm(norm)
        cb = plt.colorbar(im, ticks=np.linspace(cbar_min, cbar_max, 5))
    else:
        cb = plt.colorbar(im)
    
    return fig

####### Here we define parallel shots

def make_parallel_shots(pwrap, nsources, x_pos_sources_arr_all, z_pos_sources, x_pos_receivers_arr_all, z_pos_receivers, peakfreq):  
    min_nr_per_process = nsources / pwrap.size
    nr_leftover_processes =  nsources % (min_nr_per_process * pwrap.size)
     
    nr_shots_this_process = min_nr_per_process
    if pwrap.rank < nr_leftover_processes:
        nr_shots_this_process += 1    
    
    local_shots = []
    for i in xrange(nr_shots_this_process):
        
        all_shot_index = i*pwrap.size + pwrap.rank
        print "CREATING SHOT WITH INDEX: %i"%all_shot_index
        
        source = PointSource(m, (x_pos_sources_arr_all[all_shot_index], z_pos_sources), RickerWavelet(peakfreq), approximation='gaussian')
        
####### Here we define set of receivers

        receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation='gaussian') for x in x_pos_receivers])
    
####### Here we create and store the shots

        shot = Shot(source, receivers)
        local_shots.append(shot)

    return local_shots
    
if __name__ == '__main__':
            
            

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    print "size = %i and rank = %i"%(size, rank)

    pwrap = ParallelWrapShot(comm=comm)  
  
  

####### Here we set the wave speed

    WaveSpeed.add_lower_bound(1000.0)
    WaveSpeed.add_upper_bound(6500.0)
    
    x_lbc = PML(600.0,100.0); x_rbc = PML(600.0,100.0); z_lbc = PML(600.0,100.0); z_rbc = PML(600.0,100.0)
    C, C0, m, d = marmousi(patch='mini_square', x_lbc = x_lbc, x_rbc = x_rbc, z_lbc = z_lbc, z_rbc = z_rbc)

    n_nodes_x = m.x.n
    n_nodes_z = m.z.n
    
    dx = m.x.delta
    dz = m.z.delta
    
    x_min = d.x.lbound
    x_max = d.x.rbound
    z_min = d.z.lbound
    z_max = d.z.rbound    

    x_min_km = x_min/1000.0; x_max_km = x_max/1000.0; z_min_km = z_min/1000.0; z_max_km = z_max/1000.0 

    marmousi_baseline_true_2d    = np.reshape(C , (n_nodes_z, n_nodes_x), 'F')
    marmousi_init_2d             = np.reshape(C0, (n_nodes_z, n_nodes_x), 'F')
    
    x_label = 'Horizontal coordinate (km)'
    z_label = 'Depth (km)'
    title_marmousi_baseline_true = 'True marmousi baseline'
    title_marmousi_init = 'Initial marmousi'
    cbar_min_vel = 1500.0
    cbar_max_vel = 4600.0
    
    if rank == 0:
        fig_marmousi_baseline_true = plot_func(1, marmousi_baseline_true_2d, x_min_km, x_max_km, z_min_km, z_max_km, x_label, z_label, title_marmousi_baseline_true, cbar_min = cbar_min_vel, cbar_max = cbar_max_vel)
        fig_marmousi_init          = plot_func(2,          marmousi_init_2d, x_min_km, x_max_km, z_min_km, z_max_km, x_label, z_label,          title_marmousi_init, cbar_min = cbar_min_vel, cbar_max = cbar_max_vel)

   
    true_change_2d = np.zeros((n_nodes_z, n_nodes_x))
    layer_5_node_nr_z = int(2500.0/20.0); layer_4_node_nr_z = layer_5_node_nr_z - 1; layer_3_node_nr_z = layer_5_node_nr_z - 2; layer_2_node_nr_z = layer_5_node_nr_z - 3; layer_1_node_nr_z = layer_5_node_nr_z - 4
    layer_5_node_left = int(5660.0/20.0); layer_5_node_right = int(8180.0/20.0)
    layer_4_node_left = layer_5_node_left+1; layer_4_node_right = layer_5_node_right - 2
    layer_3_node_left = layer_4_node_left+4; layer_3_node_right = layer_4_node_right - 6
    layer_2_node_left = layer_3_node_left+3; layer_2_node_right = layer_3_node_right - 8
    layer_1_node_left = layer_2_node_left+6; layer_1_node_right = layer_2_node_right - 9
    
    true_perturb = -200.0
    true_change_2d[layer_5_node_nr_z, layer_5_node_left:layer_5_node_right] = true_perturb
    true_change_2d[layer_4_node_nr_z, layer_4_node_left:layer_4_node_right] = true_perturb
    true_change_2d[layer_3_node_nr_z, layer_3_node_left:layer_3_node_right] = true_perturb
    true_change_2d[layer_2_node_nr_z, layer_2_node_left:layer_2_node_right] = true_perturb
    true_change_2d[layer_1_node_nr_z, layer_1_node_left:layer_1_node_right] = true_perturb

    cbar_min_perturb = -np.abs(true_perturb)
    cbar_max_perturb =  np.abs(true_perturb)
    
    marmousi_monitor_true_2d = marmousi_baseline_true_2d + true_change_2d
    nsources = 19
    nreceiver = n_nodes_x 
    source_spacing = 480.0
    x_pos_sources_baseline = np.arange(0.5*source_spacing, x_max, source_spacing)
    x_pos_sources_monitor  = x_pos_sources_baseline - 240.0 
    z_pos_sources          = z_min + dz
    x_pos_receivers        = np.linspace(x_min, x_max, n_nodes_x)
    z_pos_receivers        = z_min + dz

    peakfreq = 6.0
    local_shots_baseline = make_parallel_shots(pwrap, nsources, x_pos_sources_baseline, z_pos_sources, x_pos_receivers, z_pos_receivers, peakfreq)
    local_shots_monitor  = make_parallel_shots(pwrap, nsources, x_pos_sources_monitor , z_pos_sources, x_pos_receivers, z_pos_receivers, peakfreq)
    
    trange = (0.0, 7.0)
    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')

    marmousi_init          = np.reshape(         marmousi_init_2d, (n_nodes_z*n_nodes_x, 1), 'F')
    marmousi_baseline_true = np.reshape(marmousi_baseline_true_2d, (n_nodes_z*n_nodes_x, 1), 'F')
    marmousi_monitor_true  = np.reshape( marmousi_monitor_true_2d, (n_nodes_z*n_nodes_x, 1), 'F')

    tt = time.time()
    marmousi_init_model             = solver.ModelParameters(m,{'C':  marmousi_init})
    marmousi_baseline_true_model    = solver.ModelParameters(m,{'C':  marmousi_baseline_true})
    marmousi_monitor_true_model     = solver.ModelParameters(m,{'C':  marmousi_monitor_true})
    generate_seismic_data(local_shots_baseline, solver, marmousi_baseline_true_model)
    print 'Baseline data generation: {0}s'.format(time.time()-tt)



####### Inversion algorithm
    
    objective = TemporalLeastSquares(solver, parallel_wrap_shot=pwrap)
    invalg = LBFGS(objective, memory_length=10)
    tt = time.time()

    nsteps = 30
    status_configuration = {'value_frequency'           : 1,
                     'residual_length_frequency' : 1,
                     'objective_frequency'       : 1,
                     'step_frequency'            : 1,
                     'step_length_frequency'     : 1,
                     'gradient_frequency'        : 1,
                     'gradient_length_frequency' : 1,
                     'run_time_frequency'        : 1,
                     'alpha_frequency'           : 1,
                     }

    print "backtrack linesearch is not optimal. Does not guarantee that strong wolfe conditions are satisfied. But this is used in marmousi2D example."
    line_search = 'backtrack'

    if input_marmousi_inverted_given:
        if rank == 0:
            indict = loadmat(input_marmousi_inverted)
            if 'marmousi_baseline_inverted_2d' in indict:
                marmousi_baseline_inverted_2d = indict['marmousi_baseline_inverted_2d']
            elif 'marmousi_baseline_inverted_bounded_2d' in indict:
                marmousi_baseline_inverted_2d = indict['marmousi_baseline_inverted_bounded_2d']
            else:
                raise Exception('wrong key!')
            

        else:
            marmousi_baseline_inverted_2d = None
        
    else:
        print "Starting baseline inversion to improve initial model."
    
        result = invalg(local_shots_baseline, marmousi_init_model, nsteps,
                        line_search=line_search,
                        status_configuration=status_configuration, verbose=True)
    
        print 'Run time:  {0}s'.format(time.time()-tt)
    
        if rank == 0:
            
            
            
 ###### Saving the results
    
            marmousi_baseline_inverted_2d = result.C.reshape((n_nodes_z,n_nodes_x), order='F')
    
            out = {'marmousi_baseline_inverted_bounded_2d':marmousi_baseline_inverted_2d, 'marmousi_baseline_true_2d':marmousi_baseline_true_2d.reshape((n_nodes_z,n_nodes_x), order='F')}
            savemat('baseline_inverted_bounded_nsteps_' + str(nsteps) + '.mat',out)
        
        
        else:
            marmousi_baseline_inverted_2d = None
        
    
    marmousi_baseline_inverted_2d = comm.bcast(marmousi_baseline_inverted_2d, root=0)     generate_seismic_data(local_shots_monitor, solver, marmousi_monitor_true_model)
    print 'Monitor data generation: {0}s'.format(time.time()-tt)    
    
    nswaps      = 16
    nsteps_each = 12

    beta_not_normalized_c = np.zeros((n_nodes_z *n_nodes_x, 1))
    beta_not_normalized_m = np.zeros((n_nodes_z *n_nodes_x, 1))
    
    beta_not_normalized_c_history_2d = []
    beta_not_normalized_m_history_2d = []
    marmousi_new_swap_history            = []
    
    marmousi_curr_swap          = np.reshape(marmousi_baseline_inverted_2d, (n_nodes_z*n_nodes_x,1), 'F')
    marmousi_curr_swap_m        = 1.0/marmousi_curr_swap**2
    marmousi_curr_swap_model    = solver.ModelParameters(m,{'C':  marmousi_curr_swap})

    marmousi_prev_swap          = np.zeros(marmousi_curr_swap.shape  )
    marmousi_prev_swap_m        = np.zeros(marmousi_curr_swap_m.shape)

    
    for i in xrange(nswaps):
        if rank == 0:
            print "Starting swap %i"%i
        
        if i%2 == 0:
            local_shots_cur_swap = local_shots_monitor
            if rank == 0:
                print "Using monitor shots"
            
        else:
            local_shots_cur_swap = local_shots_baseline
            if rank == 0:
                print "Using baseline shots"
    

        result = invalg(local_shots_cur_swap, marmousi_curr_swap_model, nsteps_each,
                        line_search=line_search,
                        status_configuration=status_configuration, verbose=True)
    
        
    
        marmousi_new_swap          = result.C
        marmousi_new_swap_m        = marmousi_new_swap**-2
        marmousi_new_swap_model    = solver.ModelParameters(m,{'C':  marmousi_new_swap})
    
        if i == 0:
            if rank == 0:
                out = {'marmousi_new_swap_first_iter':marmousi_new_swap}
                savemat('marmousi_new_swap_first_iter_nsteps_' + str(nsteps) + '_nswaps_' + str(nswaps) + '_nsteps_each_' + str(nsteps_each),out)
        
        if i > 0:
            diff_c_new = marmousi_new_swap - marmousi_curr_swap; diff_c_old = marmousi_curr_swap - marmousi_prev_swap;
            diff_m_new = marmousi_new_swap_m - marmousi_curr_swap_m; diff_m_old = marmousi_curr_swap_m - marmousi_prev_swap_m;
        
            beta_not_normalized_c  +=  (np.ones((n_nodes_z * n_nodes_x,1)) - np.sign(diff_c_old * diff_c_new)) * np.abs(diff_c_new)
            beta_not_normalized_m  +=  (np.ones((n_nodes_z * n_nodes_x,1)) - np.sign(diff_m_old * diff_m_new)) * np.abs(diff_m_new)
        
            beta_not_normalized_c_history_2d.append(np.copy(beta_not_normalized_c.reshape((n_nodes_z,n_nodes_x), order='F')))
            beta_not_normalized_m_history_2d.append(np.copy(beta_not_normalized_m.reshape((n_nodes_z,n_nodes_x), order='F')))
            
            marmousi_new_swap_history.append(marmousi_new_swap)

        marmousi_prev_swap          = marmousi_curr_swap
        marmousi_prev_swap_m        = marmousi_curr_swap_m
        marmousi_prev_swap_model    = marmousi_curr_swap_model
        
        marmousi_curr_swap          = marmousi_new_swap
        marmousi_curr_swap_m        = marmousi_new_swap_m
        marmousi_curr_swap_model    = marmousi_new_swap_model
        
        
        
        
    if rank == 0:

        out = {'beta_not_normalized_c_history_2d':beta_not_normalized_c_history_2d, 'beta_not_normalized_m_history_2d':beta_not_normalized_m_history_2d, 'marmousi_new_swap_history':marmousi_new_swap_history}
        savemat('beta_history_nsteps_' + str(nsteps) + '_nswaps_' + str(nswaps) + '_nsteps_each_' + str(nsteps_each) + '_marmousi_history.mat',out)
