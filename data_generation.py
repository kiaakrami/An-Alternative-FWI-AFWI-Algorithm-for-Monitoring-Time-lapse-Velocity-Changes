####### We start by importing some useful packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import sys, getopt
import copy
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
    local_shots_indices = []
    for i in xrange(nr_shots_this_process):
        
        all_shot_index = i*pwrap.size + pwrap.rank
        print "CREATING SHOT WITH INDEX: %i"%all_shot_index
        
        source = PointSource(m, (x_pos_sources_arr_all[all_shot_index], z_pos_sources), RickerWavelet(peakfreq), approximation='gaussian')
        
####### Here we define set of receivers
        receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation='gaussian') for x in x_pos_receivers])
    
####### Here we create and store the shots
        shot = Shot(source, receivers)
        local_shots.append(shot)
        local_shots_indices.append(i)

    return local_shots, local_shots_indices

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
    
    x_label = 'Horizontal coordinate (km)'
    z_label = 'Depth (km)'
    title_marm_baseline_true = 'True Marmousi baseline'
    title_marm_init = 'Initial Marmousi'
    cbar_min_vel = 1500.0
    cbar_max_vel = 4600.0

    
    nsources = 19
    nreceiver = n_nodes_x 
    source_spacing = 480.0
    x_pos_sources_baseline = np.arange(0.5*source_spacing, x_max, source_spacing)
    x_pos_sources_monitor  = x_pos_sources_baseline - 240.0 
    z_pos_sources          = z_min + dz
    x_pos_receivers        = np.linspace(x_min, x_max, n_nodes_x)
    z_pos_receivers        = z_min + dz

    
    peakfreq = 6.0
    
    local_shots_baseline, local_shots_baseline_indices = make_parallel_shots(pwrap, nsources, x_pos_sources_baseline, z_pos_sources, x_pos_receivers, z_pos_receivers, peakfreq)
    local_shots_monitor , local_shots_monitor_indices = make_parallel_shots(pwrap, nsources, x_pos_sources_monitor , z_pos_sources, x_pos_receivers, z_pos_receivers, peakfreq)
    
    trange = (0.0, 7.0)
    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=trange,
                                         kernel_implementation='cpp')

    marm_baseline_true = np.reshape(marm_baseline_true_2d, (n_nodes_z*n_nodes_x, 1), 'F')
    marm_monitor_true  = np.reshape( marm_monitor_true_2d, (n_nodes_z*n_nodes_x, 1), 'F')
    marm_baseline_initial_inverted = np.reshape(marm_baseline_initial_inverted_2d, (n_nodes_z*n_nodes_x, 1), 'F')
    
    marm_baseline_true_model             = solver.ModelParameters(m,{'C':  marm_baseline_true})
    marm_monitor_true_model              = solver.ModelParameters(m,{'C':  marm_monitor_true})
    
    marm_init_model_baseline             = solver.ModelParameters(m,{'C':  marm_baseline_initial_inverted})
    marm_init_model_monitor              = copy.deepcopy(marm_init_model_baseline)
    marm_init_model = JointModel(marm_init_model_baseline, marm_init_model_monitor)

    tt = time.time()
    generate_seismic_data(local_shots_baseline, solver, marm_baseline_true_model)
    print 'Baseline data generation: {0}s'.format(time.time()-tt)

    tt = time.time()
    generate_seismic_data(local_shots_monitor, solver, marm_monitor_true_model)
    print 'Monitor data generation: {0}s'.format(time.time()-tt)
    
####### Inversion algorithm
    
    objective = TemporalLeastSquares(solver, parallel_wrap_shot=pwrap)
    invalg_joint = LBFGS(objective, memory_length=6)     
    

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
    
    line_search = 'backtrack'
    
    if rank == 0:
        print "backtrack linesearch is not optimal"
    
    beta = np.reshape(beta_2d, (n_nodes_z*n_nodes_x,1),'F')
    beta_normalize = beta/beta.max()
    beta_min = 0.001*np.max(beta_normalize)
    beta_normalize[np.where(beta_normalize <= beta_min)] = beta_min
    model_reg_term_scale = 1e4

    
    
    result = invalg_joint(local_shots_baseline, local_shots_monitor, beta_normalize, model_reg_term_scale, marm_init_model, nsteps,line_search=line_search,status_configuration=status_configuration, verbose=True)
    result_baseline_model = result.m_0
    result_monitor_model  = result.m_1
    if rank == 0:
 
 
 ###### Saving the results
 
        marm_baseline_inverted_2d = result_baseline_model.C.reshape((n_nodes_z,n_nodes_x), order='F')
        marm_monitor_inverted_2d  = result_monitor_model.C.reshape( (n_nodes_z,n_nodes_x), order='F')
        marm_diff_inverted_2d = marm_monitor_inverted_2d - marm_baseline_inverted_2d 
    
        out = {'marm_baseline_inverted_bounded_2d':marm_baseline_inverted_2d, 'marm_monitor_inverted_bounded_2d':marm_monitor_inverted_2d, 'marm_diff_inverted_bounded_2d':marm_diff_inverted_2d}
        savemat('joint_output_' + str(nsteps) + '_model_reg_term_scale_' + str(model_reg_term_scale) + '_beta_min_' + str(beta_min) + '.mat',out)
        print '...run time:  {0}s'.format(time.time()-tt)


        

    
        
    
        
