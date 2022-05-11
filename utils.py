################################################
############ HELPER FUNCTIONS ##################
################################################
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import h5py
import os
import glob
import numpy as np
from tqdm import trange, tqdm

# 
def remove_duplicate_states(data):
    
    data2 = []
    data2.append(data[0])
    for k in range(1, data.shape[0], 1):
        
        if (data[k]-data[k-1]).sum(0)>1E-10:
            data2.append(data[k])
            
    data2 = np.array(data2)
    
    return data2

#
def run_pca(data):
    from sklearn.decomposition import PCA
    pca = PCA(svd_solver='full')
    X_pca = pca.fit_transform(data)
    print ("   Initial X_pca: ", X_pca.shape)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    #print ("cumsum: ", cumsum)
    idx = np.where(cumsum>0.95)[0]
    
    print ("   # of comps to 95% exp variance: ", idx[0])
    return X_pca[:,:idx[0]]


def run_pca_return_pca(data):
    from sklearn.decomposition import PCA
    pca = PCA(svd_solver='full')
    X_pca = pca.fit_transform(data)
    print ("   Initial X_pca: ", X_pca.shape)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    #print ("cumsum: ", cumsum)
    idx = np.where(cumsum>0.95)[0]
    
    print ("   # of comps to 95% exp variance: ", idx[0])
    return pca, X_pca[:,:idx[0]]


#
def run_deduplication(X_pca):
    
    print ("pre duplcitaion: ", X_pca.shape)
    X_pca2 = remove_duplicate_states(X_pca)
    print ("X_pca2: ", X_pca2.shape)
    print (" % of unique states: ", round(X_pca2.shape[0]/X_pca.shape[0],2))

    return X_pca2

#
def run_addnoise(X_pca2):
    
    # add noise to prevent crashes
    noise = np.random.rand(X_pca2.shape[0], 
                           X_pca2.shape[1])*1E-3
    X_pca2 = X_pca2+noise

    return X_pca2

#
def run_binning(data, bin_size=7, sum_flag=True):
    
    # split data into bins
    idx = np.arange(0, data.shape[0], bin_size)
    d2 = np.array_split(data, idx[1:])
    
    # sum on time axis; drop last value
    if sum_flag:
        d3 = np.array(d2[:-1]).sum(1)
    else:
        d3 = np.median(np.array(d2[:-1]), axis=1)
    
    print ("   Data binned using ", bin_size, " frame bins, final size:  ", d3.shape)

    #    
    return d3

#
def knn_triage_step(pca_wf, triage_value):

    knn_triage_threshold = 100*(1-triage_value)

    if pca_wf.shape[0] > 1/triage_value:
        idx_keep = knn_triage(knn_triage_threshold, pca_wf)
        idx_keep = np.where(idx_keep==1)[0]
    else:
        idx_keep = np.arange(pca_wf.shape[0])

    return idx_keep  
 
#
def knn_triage(th, pca_wf):

    tree = cKDTree(pca_wf)
    dist, ind = tree.query(pca_wf, k=6)
    dist = np.sum(dist, 1)

    idx_keep1 = dist <= np.percentile(dist, th)
    return idx_keep1

#
def load_tracks_nathalie(fname_tracks):

    #
    pos_tracks = []
    idx_tracks = []
    idx_ctr = 0
    for fname_track in fname_tracks:
        with h5py.File(fname_track, 'r') as file:
            #
            pos = file['trdEval']['position_atframe'][()].squeeze()

            n_timesteps = pos.shape[0]
            print ("# of time steps: ", pos.shape)

        # bin speed for some # of bins
        bin_width = 5
        sum_flag = False
        pos_bin = run_binning(pos, bin_width, sum_flag)
        print ("pos bin: ", pos_bin.shape)

        #
        pos_tracks.append(pos_bin)

        # add locations of the belt realtive
        temp = np.arange(idx_ctr,idx_ctr+n_timesteps,1)
        idx_tracks.append(temp)
    
        idx_ctr+=n_timesteps
        
    return pos_tracks, idx_tracks



#
def load_tracks(fname_tracks):

    #
    pos_tracks = []
    idx_tracks = []
    idx_ctr = 0
    for fname_track in fname_tracks:
        with h5py.File(fname_track, 'r') as file:
            #
            pos = file['trdEval']['position_atframe'][()].squeeze()

            n_timesteps = pos.shape[0]
            print ("# of time steps: ", pos.shape)

        # bin speed for some # of bins
        bin_width = 5
        sum_flag = False
        pos_bin = run_binning(pos, bin_width, sum_flag)
        print ("pos bin: ", pos_bin.shape)

        #
        pos_tracks.append(pos_bin)

        # add locations of the belt realtive
        temp = np.arange(idx_ctr,idx_ctr+n_timesteps,1)
        idx_tracks.append(temp)
    
        idx_ctr+=n_timesteps
        
    return pos_tracks, idx_tracks

# 

def load_ca_bin(fname_ca, idx_track, bin_width):
    data = np.load(fname_ca).T
    print ("All [ca] data: ", data.shape)

    # select data to analyzie
    data = data[idx_track]
    print ("   Specific belt data: ", data.shape)


    # 
    data = run_binning(data, bin_width, sum_flag=True)

    #
   
    return data
    

def load_ca_bin_pca(fname_ca, idx_track, bin_width):

    data = np.load(fname_ca).T
    print ("All [ca] data: ", data.shape)

    # select data to analyzie
    data = data[idx_track]
    print ("   Specific belt data: ", data.shape)


    # 
    data = run_binning(data, bin_width, sum_flag=True)

    #
    X_pca = run_pca(data)

    #
    print ("Final data (n_timesteps, n_dimensions): ", X_pca.shape)
    
    return X_pca
    
#
def run_isomap(session_ids, fname_tracks,bin_width):
    for session_id,fname_track in zip(session_ids,fname_tracks):
        idx_track = idx_tracks[session_id]

        #
        X_pca = load_ca_bin_pca(fname_ca, idx_track, bin_width)

        # 
        fname_out = fname_track.replace('.mat','_'+str(bin_width)+"bin_isomap.npy")
        if os.path.exists(fname_out)==False:
            print ("Running Isomap: ", X_pca.shape)
            embedding = Isomap(n_components=3, n_jobs = -1)
            X_isomap = embedding.fit_transform(X_pca)
            np.save(fname_out, X_isomap)

        else:
            X_isomap = np.load(fname_out)

        print ("")
        
    print ("    DONE .... ")
    
    return None
 
def plot_2d(X_pca, pos_track, ax1, title, n_segs, alpha=1):
    
    # 
    cmap = plt.get_cmap('viridis', n_segs)
#     print ("pos track: ", np.unique(pos_track))
#     print ('median pos_track:', np.median(pos_track))
    #
    p = ax1.scatter(X_pca[:,0],
                 X_pca[:,1],
                 #X_pca[:,2],
                 c=cmap(pos_track),
                 #cmap='viridis',
                    
                 #cmap = pos_track,
                 #edgecolor='black',
                 alpha=alpha)
    
    #ax=plt.gca() #get the current axes
#     PCM=ax1.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
#     plt.colorbar(PCM, ax=ax1) 

    plt.title(title)
    
    return p



def plot_3d(X_pca, pos_track, ax1, title, n_segs, alpha=1):
    
    # 
    cmap = plt.get_cmap('viridis', n_segs)
#     print ("pos track: ", np.unique(pos_track))
#     print ('median pos_track:', np.median(pos_track))
    #
    p = ax1.scatter3D(X_pca[:,0],
                 X_pca[:,1],
                 X_pca[:,2],
                 c=cmap(pos_track),
                 #cmap='viridis',
                    
                 #cmap = pos_track,
                 #edgecolor='black',
                 alpha=alpha)
    
    #ax=plt.gca() #get the current axes
#     PCM=ax1.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
#     plt.colorbar(PCM, ax=ax1) 

    #plt.title(title)
    
    return p

#
def plot_3d_distributions(fig, fname_in, X_pca, triage_value, pos_track
                          ):
    
    # 
    fname_triaged_idx = fname_in.replace('.mat','_'+str(triage_value)+'.npy')
    
    # 
    if os.path.exists(fname_triaged_idx)==False:
        # 
        if triage_value!=0:
            idx_pca = knn_triage_step(X_pca, triage_value)
            np.save(fname_triaged_idx, idx_pca)
        else:
            idx_pca = np.arange(X_pca.shape[0])
    
    else:
        # 
        if triage_value!=0:
            idx_pca = np.load(fname_triaged_idx)
        else:
            idx_pca = np.arange(X_pca.shape[0])

            
    #
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 
    n_segs = np.unique(pos_track).shape[0]
    print ("Data to plot: ", X_pca[idx_pca].shape, pos_track[idx_pca].shape)
    pall = plot_3d(X_pca[idx_pca], pos_track[idx_pca], ax1, 'pca -all', n_segs,
                   alpha=.3)
    
    # 
    #cbar = fig.colorbar(pall, ax=ax1)
    cbar = fig.colorbar(pall, ax=ax1, ticks=np.arange(0,181,10)/180.)
    yticks = np.arange(0,181,10)
    cbar.ax.set_yticklabels(yticks)  # vertically oriented colorbar

    cbar.set_label('Belt position (cm)', size=18)

    # Plot track segment distributions
    if False: 
        fig = plt.figure()
    #    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

        min_ = np.min(X_pca,axis=0)
        max_ = np.max(X_pca,axis=0)
        #print (min_, max_)

        for k in range(np.unique(pos_track).shape[0]):
            try:
                ax1 = fig.add_subplot(4, 5, k+1, projection='3d')
            except:
                print (" out of plotting panels")
                break

            #plot_3d(X_pca[idx_pca], pos_track[idx_pca], ax1, 'pca -all',0.01)

            #idx_seg = np.where(pos_track[idx_pca]==clrs[k])[0]
            idx_seg = np.where(pos_track[idx_pca]==k)[0]

            p = plot_3d(X_pca[idx_pca][idx_seg], 
                        pos_track[idx_pca][idx_seg], 
                        ax1, 
                        'pca - track segment '+str(k),
                       n_segs,
                       alpha=.5)

            plt.xlim(min_[0],max_[0])    
            plt.ylim(min_[1],max_[1])    
            #ax1.set_zlim(min_[2],max_[2])    

    #return pall
    
    return X_pca[idx_pca], pos_track[idx_pca]

#
def run_umap(session_ids, fname_tracks,bin_width):
    import umap
    from sklearn.datasets import load_digits

    # 
    for session_id,fname_track in zip(session_ids,fname_tracks):
        idx_track = idx_tracks[session_id]

        #
        X_pca = load_ca_bin_pca(fname_ca, idx_track, bin_width)

        # 
        fname_out = fname_track.replace('.mat','_'+str(bin_width)+"bin_umap.npy")
        if os.path.exists(fname_out)==False:
            print ("Running Umap: ", X_pca.shape)
            
            X_umap = umap.UMAP().fit_transform(X_pca)
            
            np.save(fname_out, X_umap)

        else:
            X_umap = np.load(fname_out)

        print ("")
        
    print ("    DONE .... ")
    
    
def plot_2d_distributions(X_pca, triage_value, pos_track
                          ):
    
    if triage_value==0:
        idx_pca = knn_triage_step(X_pca, triage_value)
    else:
        idx_pca = np.arange(X_pca.shape[0])
    
    #
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    
    n_segs = np.unique(pos_track).shape[0]
    pall = plot_2d(X_pca[idx_pca], pos_track[idx_pca], ax1, 'pca -all', n_segs,
                   alpha=.3)
    
    # # 
    #cbar = fig.colorbar(pall, ax=ax1)
    cbar = fig.colorbar(pall, ax=ax1, ticks=np.arange(0,181,10)/180.)
    yticks = np.arange(0,181,10)
    cbar.ax.set_yticklabels(yticks)  # vertically oriented colorbar

    cbar.set_label('Belt position (cm)', size=18)

    # keep same shape plot
    fig = plt.figure()
#    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    min_ = np.min(X_pca,axis=0)
    max_ = np.max(X_pca,axis=0)
    print (min_, max_)

    for k in range(np.unique(pos_track).shape[0]):
        ax1 = fig.add_subplot(4, 5, k+1)

        #plot_3d(X_pca[idx_pca], pos_track[idx_pca], ax1, 'pca -all',0.01)
        
        #idx_seg = np.where(pos_track[idx_pca]==clrs[k])[0]
        idx_seg = np.where(pos_track[idx_pca]==k)[0]
        
        p = plot_2d(X_pca[idx_pca][idx_seg], 
                    pos_track[idx_pca][idx_seg], 
                    ax1, 
                    'pca - track segment '+str(k),
                   n_segs,
                   alpha=.5)

        plt.xlim(min_[0],max_[0])    
        plt.ylim(min_[1],max_[1])    
        #ax1.set_zlim(min_[2],max_[2])    
        
    return pall

def split_track( 
                seg_len, 
                pos_track_triaged,
                X_pca_triaged):
    track_segs = np.arange(0, 180, seg_len)
    loc_idx = []
    for seg in track_segs:
        idx = np.where(np.logical_and(pos_track_triaged>=seg,
                                      pos_track_triaged<(seg+seg_len)))[0]
        loc_idx.append(idx)

    # pca-ave
    pca_ave = []
    pca_std = []
    for k in range(len(loc_idx)):
        temp = X_pca_triaged[loc_idx[k]]
        pca_std.append(np.std(temp,axis=0)[:3])

        #
        temp = np.median(temp, axis=0)
        pca_ave.append(temp[:3])

    pca_ave = np.vstack(pca_ave)
    pca_std = np.vstack(pca_std)
    print (pca_ave.shape, pca_std.shape)

    return pca_ave, pca_std



def plot_segment_averages(fig, 
                          pca_ave,
                         ):

    #fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')


    for i in range(1, pca_ave.shape[0]):
        ax1.plot3D(pca_ave[i-1:i+1,0], 
                  pca_ave[i-1:i+1,1], 
                  pca_ave[i-1:i+1,2], 
                  color=plt.cm.viridis((i)/pca_ave.shape[0])
               )

    ax1.plot3D([pca_ave[0,0], pca_ave[-1,0]],
              [pca_ave[0,1], pca_ave[-1,1]],
              [pca_ave[0,2], pca_ave[-1,2]],
              color=plt.cm.viridis(0)
           )

    norm = matplotlib.colors.Normalize(vmin=0,vmax=180)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(0,180, 
                                       7
                                      ), 
                )
        
def triage(fname_track, triage_value, X_pca):
    
    #
    fname_triaged_idx = fname_track.replace('.mat','_'+str(triage_value)+'.npy')
    
    # 
    if os.path.exists(fname_triaged_idx)==False:
        # 
        if triage_value!=0:
            idx_pca = knn_triage_step(X_pca, triage_value)
        else:
            idx_pca = np.arange(X_pca.shape[0])
    
        np.save(fname_triaged_idx, idx_pca)
    else:
        #print ('loading indexes from file: ', fname_triaged_idx)
        idx_pca = np.load(fname_triaged_idx)
        
    return idx_pca

def split_track2(dimensionality,
                seg_len, 
                pos_track_triaged,
                X_pca_triaged):
    
    # 
    track_segs = np.arange(0, 180, seg_len)
    loc_idx = []
    for seg in track_segs:
        idx = np.where(np.logical_and(pos_track_triaged>=seg,
                                      pos_track_triaged<(seg+seg_len)))[0]
        loc_idx.append(idx)

    # pca-ave
    pca_ave = []
    pca_std = []
    for k in range(len(loc_idx)):
        temp = X_pca_triaged[loc_idx[k]]
        pca_std.append(np.std(temp,axis=0)[:dimensionality])

        #
        temp = np.median(temp, axis=0)
        pca_ave.append(temp[:dimensionality])

    pca_ave = np.vstack(pca_ave)
    pca_std = np.vstack(pca_std)
    print (pca_ave.shape, pca_std.shape)

    return pca_ave, pca_std

# 

from matplotlib import pyplot as plt, animation

def compute_networkx_custom(transition_matrix):

    T = transition_matrix.copy()
    np.fill_diagonal(T, 0)

    # 
    thresh = 1
    idx0 = np.where(T<thresh)
    T[idx0]=0
    idx1 = np.where(T>=thresh)
    T[idx1]=1

    #
    G = nx.from_numpy_matrix(T)
    # G = nx.DiGraph(T)
    idx = list(nx.isolates(G))
    G.remove_nodes_from(idx)
    # G.remove_nodes_from(idx)

    #pos = nx.circular_layout(G)
    #pos = nx.spring_layout(G)

    return G, T

def reload_names():
    fname_tracks = [

        ['/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S1-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S2-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/mat/DON-004366_20210228_TRD-2P_S3-ACQ_eval.mat'],

         # 
         ['/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S1-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S2-ACQ_eval.mat',
         '/media/cat/4TB/donato/steffen/DON-004366/20210301/mat/DON-004366_20210301_TRD-2P_S3-ACQ_eval.mat'],

        # 
        ['/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S1-ACQ_eval.mat',
        '/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S2-ACQ_eval.mat',
        '/media/cat/4TB/donato/steffen/DON-004366/20210303/DON-004366_20210303_TRD-2P_S3-ACQ_eval.mat']

    ]

    # 
    fnames_ca = [
        '/media/cat/4TB/donato/steffen/DON-004366/20210228/suite2p/plane0/binarized_traces/F_upphase.npy',
        '/media/cat/4TB/donato/steffen/DON-004366/20210301/binarized_traces/F_upphase.npy',
        '/media/cat/4TB/donato/steffen/DON-004366/20210303/binarized_traces/F_upphase.npy']

    # segmaentaiton: 
    segment_order = [
    ["A", "B", "A'"],
    ["A", "A'","Bplt.clf()"],
    ["A", "B", "A'"]       
    ]
    
    return fname_tracks, fnames_ca, segment_order

   
def plot_networkx(transition_matrix,
                 dimensionality):
    import networkx as nx

    T = transition_matrix.copy()
    np.fill_diagonal(T, 0)

    # 
    thresh = 1
    idx = np.where(T<thresh)
    T[idx]=0
    idx = np.where(T>=thresh)
    T[idx]=1

    #
    G = nx.from_numpy_matrix(T)
    # G = nx.DiGraph(T)
    idx = list(nx.isolates(G))
    G.remove_nodes_from(idx)
    # G.remove_nodes_from(idx)

    #pos = nx.circular_layout(G)
    pos = nx.spring_layout(G)

    # G = nx.cycle_graph(24)
    # nx.draw(G, pos, node_color=range(24), node_size=800, cmap=plt.cm.Blues)
    # plt.show()

    #
    plt.figure()
    nx.draw(G, 
            pos, 
            node_color=range(T.shape[0]-len(idx)), 
            node_size=100, 
                cmap=plt.cm.viridis)

#
def plot_networkx_3D(transition_matrix,
                 dimensionality):
    import networkx as nx

    T = transition_matrix.copy()
    np.fill_diagonal(T, 0)

    # 
    thresh = 1
    idx = np.where(T<thresh)
    T[idx]=0
    idx = np.where(T>=thresh)
    T[idx]=1

    #
    G = nx.from_numpy_matrix(T)
    # G = nx.DiGraph(T)
    idx = list(nx.isolates(G))
    G.remove_nodes_from(idx)
    # G.remove_nodes_from(idx)

    #pos = nx.circular_layout(G)
      
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3, seed=779)
    
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, c=np.arange(node_xyz.shape[0]),
               ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


    _format_axes(ax)
    fig.tight_layout()
    #plt.show()
    
    
    
def point_on_circle(angle):
    '''
        Finding the x,y coordinates on circle, based on given angle
    '''
    from math import cos, sin, pi
    #center of circle, angle in degree and radius of circle
    center = [0,0]
    #angle = pi / 2
    radius = 100
    x = center[0] + (radius * cos(angle))
    y = center[1] + (radius * sin(angle))

    return x,y


    
def compute_transition_matrix(idx_tracks,
                             pos_tracks,
                             fname_tracks1,
                             session_id,
                             leg_seg_cm,
                             triage_value, 
                             randomize=False):

    # 
    idx_track = idx_tracks[session_id]
    pos_track = pos_tracks[session_id]           
    fname_track = fname_tracks1[session_id]
    pos_track = np.int32(pos_track//len_seg_cm)

    #
    if False:
        data = load_ca_bin(fname_ca, idx_track, bin_width)
        print ("data: ", data.shape)
        print (np.unique(data,axis=0).shape)
    else:
        data = load_ca_bin_pca(fname_ca1, idx_track, bin_width)
        print ("data: ", data.shape)
        print (np.unique(data,axis=0).shape)

    dimensionality = data.shape[1]
    #dimensionality = 100

    # remove outliers
    seg_len = 1  # 1cm chunks
    idx_triaged = triage(fname_track, triage_value, data)

    #
    #randomize=False
    if randomize:
        idx_triaged = np.random.choice(idx_triaged, idx_triaged.shape[0], replace=False)

    # remove outliers
    pos_track_triaged = pos_track[idx_triaged]
    data_triaged = data[idx_triaged]
    
    # compute the averate population vector at each 1cm location on the belt
    pca_ave, pca_std = split_track2(dimensionality,
                                    seg_len, 
                                    pos_track_triaged,
                                    data_triaged)

    # we now have the average neural activity representing eachlocation in space:
    #  will build transition matrix for each neural state occuring at time t
    #    1) identify which location on track is nearest at time=t
    #    2) identify which location on track is nearest to time=t+1

    transition_matrix = np.zeros((180,180))
    neural_location = np.zeros(data_triaged.shape[0],dtype='int32')
    for k in trange(data_triaged.shape[0]-1):

        # find location of current state in PCA-space
        temp_state = data_triaged[k,:dimensionality]  # grab only first 3 components, but ok to grab more
        diff = np.abs(pca_ave-temp_state).sum(axis=1)
        idx1 = np.nanargmin(diff)
        neural_location[k] = idx1

        # find location of the next state in PCA-space
        temp_state = data_triaged[k+1,:dimensionality]  # grab only first 3 components, but ok to grab more
        diff = np.abs(pca_ave-temp_state).sum(axis=1)
        idx2 = np.nanargmin(diff)

        # link them
        transition_matrix[idx1,idx2]+=1

    return transition_matrix, neural_location

#
def compute_transition_matrix_spontaneous(idx_tracks,
                                          pos_tracks,
                                          fname_tracks1,
                                          session_id,
                                          leg_seg_cm,
                                          triage_value, 
                                          randomize=False):

    # 
    idx_track = idx_tracks[session_id]
    pos_track = pos_tracks[session_id]           
    fname_track = fname_tracks1[session_id]
    pos_track = np.int32(pos_track//len_seg_cm)

    #
    if False:
        data = load_ca_bin(fname_ca, idx_track, bin_width)
        print ("data: ", data.shape)
        print (np.unique(data,axis=0).shape)
    else:
        data = load_ca_bin_pca(fname_ca1, idx_track, bin_width)
        print ("data: ", data.shape)
        print (np.unique(data,axis=0).shape)

    dimensionality = data.shape[1]
    #dimensionality = 100

    # remove outliers
    seg_len = 1  # 1cm chunks
    idx_triaged = triage(fname_track, triage_value, data)

    #
    #randomize=False
    if randomize:
        idx_triaged = np.random.choice(idx_triaged, idx_triaged.shape[0], replace=False)

    # remove outliers
    pos_track_triaged = pos_track[idx_triaged]
    data_triaged = data[idx_triaged]
    
    # compute the averate population vector at each 1cm location on the belt
    pca_ave, pca_std = split_track2(dimensionality,
                                    seg_len, 
                                    pos_track_triaged,
                                    data_triaged)

    # we now have the average neural activity representing eachlocation in space:
    #  will build transition matrix for each neural state occuring at time t
    #    1) identify which location on track is nearest at time=t
    #    2) identify which location on track is nearest to time=t+1

    transition_matrix = np.zeros((180,180))
    neural_location = np.zeros(data_triaged.shape[0],dtype='int32')
    for k in trange(data_triaged.shape[0]-1):

        # find location of current state in PCA-space
        temp_state = data_triaged[k,:dimensionality]  # grab only first 3 components, but ok to grab more
        diff = np.abs(pca_ave-temp_state).sum(axis=1)
        idx1 = np.nanargmin(diff)
        neural_location[k] = idx1

        # find location of the next state in PCA-space
        temp_state = data_triaged[k+1,:dimensionality]  # grab only first 3 components, but ok to grab more
        diff = np.abs(pca_ave-temp_state).sum(axis=1)
        idx2 = np.nanargmin(diff)

        # link them
        transition_matrix[idx1,idx2]+=1

    return transition_matrix, neural_location

def load_tracks_nathalie(fname_locs,
                         fname_bin,
                         fname_start):


    #
    d = np.load(fname_bin,
                  allow_pickle=True)
    F_upphase = d["F_upphase"]
    print ("bin upphase: ", F_upphase.shape)

    # start of recording as some imaging didn't start same time as DLC tracking
    start = int(np.loadtxt(fname_start))
    print ("start: ", start)

    #
    locs = np.load(fname_locs)

    print ("All DLC locations: ", locs.shape)
    locs = locs[start:]
    print ("Final locations shape: ", locs.shape)


    return (F_upphase, locs)


# 
def get_data(fname_locs,
             fname_bin,
             fname_start):

    F_upphase, locs = load_tracks_nathalie(fname_locs,
                                           fname_bin,
                                           fname_start)
    print ("locs: ", locs.shape)
    # partition space into specific parcels
    F_upphase = F_upphase.T
    print ("F: ", F_upphase.shape)

    # bin data
    bin_size = 7
    run_binnflag = False
    if run_binnflag:
        sum_flag = True
        f_binned= run_binning(F_upphase, bin_size, sum_flag)
    else:
        f_binned = F_upphase
    print (f_binned.shape)

    # 
    if run_binnflag:
        sum_flag = False
        locs_binned = run_binning(locs, bin_size, sum_flag)
    else:
        locs_binned = locs
    print ("locs_binned: ", locs_binned.shape)


    # partition space 
    partition_size = 10               # partition size in centimeters
    box_width, box_length = 80, 80    # length in centimeters

    # centre to 0,0
    locs[:,0] = locs[:,0]-np.min(locs[:,0])
    locs[:,1] = locs[:,1]-np.min(locs[:,1])

    xmax = np.max(locs[:,0])
    ymax = np.max(locs[:,1])
    print (xmax,ymax)
    pixels_per_cm = xmax / box_width

    # convert DLC coords to cm
    locs_cm = locs_binned/pixels_per_cm
    print ("locs_cm: ", locs_cm.shape)

    # partiiont the square box;  # it will be a square array
    # it just holds the numbers 1..64 for rexample, for an 8 x 8 partition
    partition = np.zeros((box_width//partition_size, 
                          box_width//partition_size),
                          )

    # it contains matching of pop vectors to a specific bin (ranging from 1..64)
    locs_partitioned = np.zeros(locs_cm.shape[0])

    # list of 64 lists containing the population vector identities for each bin location
    partition_times = []

    ctrx = 0
    bin_ctr = 0
    for x in range(0,box_width,partition_size):
        idx = np.where(np.logical_and(locs_cm[:,0]>=x,
                                      locs_cm[:,0]<x+partition_size))[0]
        ctry = 0
        for y in range(0,box_width,partition_size):
            idy = np.where(np.logical_and(locs_cm[:,1]>=y,
                                          locs_cm[:,1]<y+partition_size))[0]

            # find intersection of idx and idy as the times when the mouse was in 
            #   partition bin: ctrx, ctry
            idx3 = np.intersect1d(idx, idy)
            #print (idx3)
            locs_partitioned[idx3]=bin_ctr

            partition_times.append(idx3)    

            #
            partition[ctry,ctrx]=bin_ctr

            #
            ctry+=1
            bin_ctr+=1

        ctrx+=1


    print ("DONE")
    
    return f_binned, locs_partitioned, partition_times, partition, box_width, box_length, locs_cm,partition_size


#
def convert_dlc(fname_csv):
    
    def load_csv(fname_csv):

        from numpy import genfromtxt
        locs = genfromtxt(fname_csv, 
                          delimiter=',', 
                          dtype='str')

        print ("Note... saving all locations regardless of likelihood...")
        print (locs.shape)

        return locs


    locs = load_csv(fname_csv)
    body_feature_idx = 5*3+1
    print ("Body feature index", np.arange(body_feature_idx,body_feature_idx+2,1))
    neck = np.float32(locs[3:,body_feature_idx:body_feature_idx+2])
    print (neck.shape)

    np.save(fname_csv[:-4]+'_locs.npy', neck)

    
    
def compute_transition_matrix_2D_exploration_data(partition_times,
                                                  data,
                                                  dimensionality):

    # can remove the weirdest outliers in the data which might help with decoding
    if False:
        # remove outliers
        idx_triaged = triage(fname_track, triage_value, data)

        #
        #randomize=False
        if randomize:
            idx_triaged = np.random.choice(idx_triaged, idx_triaged.shape[0], replace=False)

        # remove outliers
        pos_track_triaged = pos_track[idx_triaged]
        data_triaged = data[idx_triaged]
    #
    else:
        data_triaged = data

    # Can compress the data to remove noise cells etc.
    if False:
        # compute the averate population vector at each 1cm location on the belt
        #x_pca = run_pca(f_binned)
        pca, data_triaged_pca = run_pca_return_pca(data_triaged)
        print ("x_pca: ", data_triaged_pca.shape)
    else:
        data_triaged_pca = data_triaged
        
    # reduce dimensionality by taking only the first X cells or PCs
    # for now we should leave all the dimensions in the data
    data_triaged_pca = data_triaged_pca[:,:dimensionality]
    
    # compute average neural state at each spatial location
    # this can be done on all data or PCA version up to some dimensionality
    ave_states = []
    for k in range(len(partition_times)):
        temp = data_triaged_pca[partition_times[k]]
        #print ("temp: ", temp.shape)
        ave = np.nanmean(temp,axis=0)
        ave_states.append(ave)
        
    data_triaged_ave = np.vstack(ave_states)
    print ("average states: ", data_triaged_ave.shape)

	# The rest of this is for network/graph analysis
    # we now have the average neural activity representing eachlocation in space:
    #  will build transition matrix for each neural state occuring at time t
    #    1) identify which location on track is nearest at time=t
    #    2) identify which location on track is nearest to time=t+1

    transition_matrix = np.zeros((len(data_triaged_ave),
                                  len(data_triaged_ave)))
    
    # 
    neural_location = np.zeros(data_triaged.shape[0],dtype='int32')
    print ("transitoin matrix: ", transition_matrix.shape, "  neural location: ", neural_location.shape)
    print ("data_triaged pca: ", data_triaged_pca.shape, " data triaged average: ", data_triaged_ave.shape)
    print ("data tiraged average: ", data_triaged_ave)
    for k in trange(data_triaged.shape[0]-1):

        # find location of current state in PCA-space
        temp_state = data_triaged_pca[k,:dimensionality]  # grab only first n comps
        diff = np.abs(data_triaged_ave-temp_state).sum(axis=1)
        idx1 = np.nanargmin(diff)
        neural_location[k] = idx1

        # find location of the next state in PCA-space
        temp_state = data_triaged_pca[k+1,:dimensionality]  # grab only first 3 components, but ok to grab more
        diff = np.abs(data_triaged_ave-temp_state).sum(axis=1)
        idx2 = np.nanargmin(diff)

        # link them
        transition_matrix[idx1,idx2]=1

    return transition_matrix, neural_location
    
#
def compute_networkx_custom(transition_matrix):

    T = transition_matrix.copy()
    np.fill_diagonal(T, 0)

    # 
    thresh = 1
    idx0 = np.where(T<thresh)
    T[idx0]=0
    idx1 = np.where(T>=thresh)
    T[idx1]=1

    #
    G = nx.from_numpy_matrix(T)
    # G = nx.DiGraph(T)
    idx = list(nx.isolates(G))
    #G.remove_nodes_from(idx)
    # G.remove_nodes_from(idx)

    #pos = nx.circular_layout(G)
    #pos = nx.spring_layout(G)

    return G, T

#
def animate_2D(frame):
    global ctr, fig, ax1, ax2, sizes2

    #
    plt.suptitle("Time: "+str(round(ctr/sample_rate,2)).zfill(2))

    ########### TRACK SPACE #########
    ax1.clear()
    ax1=plt.subplot(1,2,1)

    #sizes = np.zeros(res.shape[0])+25
    #idx = pos_track[ctr]
    #idx = np.random.choice(np.arange(180),1)
    #sizes[idx]=250
    #ax1.imshow(partition)
    draw_grid(ax1)

    x = locs_cm[ctr,0]/partition_size
    y = locs_cm[ctr,1]/partition_size
    ax1.scatter(x,
                y,
                c='blue',
                s=200)
    ax1.set_title("Mouse location x,y "+str(round(x*partition_size,1))+ " "+str(round(y*partition_size,1)) +
                 " (cm)")

    ax1.set_xlim(0,box_width//partition_size)
    ax1.set_ylim(0,box_length//partition_size)

    ax1.set_xticks([])
    ax1.set_yticks([])

    ############################################
    ########### NEURAL STATE SPACE #############
    ############################################
    ax2.clear()
    ax2=plt.subplot(1,2,2)
    ax2.set_title("Neural state inferred location")
    #sizes = np.zeros(G.number_of_nodes())+25

    draw_grid(ax2)
    temp = neural_location[ctr]
    x = temp//(box_width//partition_size)+0.5
    y = temp%(box_length//partition_size)+0.5
    
    #
    ax2.scatter(x,
                y,
                c='red',
                s=200)
    plt.xlim(0,box_width//partition_size)
    plt.ylim(0,box_length//partition_size)

    ax2.set_xticks([])
    ax2.set_yticks([])
    
#         idx = neural_location[ctr]
#         sizes2[idx]=500

#         sizes2= sizes2-70

#         idx = np.where(sizes2<50)[0]
#         sizes2[idx]=50
#         #print ("sizes2: ", sizes2)


    if ctr%50==0:
        print ("ctr: ", ctr)

    #
    ctr+=1

    
def animate_2D_sameaxis(frame):
    global ctr, fig, ax1, ax2, sizes2

    #
    plt.suptitle("Time: "+str(round(ctr/sample_rate,2)).zfill(2))

    ########### TRACK SPACE #########
    ax1.clear()
    ax1=plt.subplot(1,1,1)

    #sizes = np.zeros(res.shape[0])+25
    #idx = pos_track[ctr]
    #idx = np.random.choice(np.arange(180),1)
    #sizes[idx]=250
    #ax1.imshow(partition)
    draw_grid(ax1)

    x = locs_cm[ctr,0]/partition_size
    y = locs_cm[ctr,1]/partition_size
    ax1.scatter(x,
                y,
                c='blue',
                s=200)
    ax1.set_title("Mouse location x,y "+str(round(x*partition_size,1))+ " "+str(round(y*partition_size,1)) +
                 " (cm)")

    ax1.set_xlim(0,box_width//partition_size)
    ax1.set_ylim(0,box_length//partition_size)

    ax1.set_xticks([])
    ax1.set_yticks([])

    ############################################
    ########### NEURAL STATE SPACE #############
    ############################################
#     ax2.clear()
#     ax2=plt.subplot(1,2,2)
#     ax2.set_title("Neural state inferred location")
#     #sizes = np.zeros(G.number_of_nodes())+25

#     draw_grid(ax2)
    temp = neural_location[ctr]
    x = temp//(box_width//partition_size)+0.5
    y = temp%(box_length//partition_size)+0.5
    
    #
    ax1.scatter(x,
                y,
                c='red',
                s=200)
    #plt.xlim(0,box_width//partition_size)
    #plt.ylim(0,box_length//partition_size)

    #ax2.set_xticks([])
    #ax2.set_yticks([])
    
#         idx = neural_location[ctr]
#         sizes2[idx]=500

#         sizes2= sizes2-70

#         idx = np.where(sizes2<50)[0]
#         sizes2[idx]=50
#         #print ("sizes2: ", sizes2)


    if ctr%50==0:
        print ("ctr: ", ctr)

    #
    ctr+=1
    
def draw_grid(ax):
    global ctr, steps, fig, ax1, ax2, sizes2,partition, sample_rate, box_width, box_length, partition_size

    width = 10
    for k in range(0,box_width//partition_size+1,1):
        ax.plot([k,k],
                 [0,box_width//partition_size],
                 linewidth = 2,
                 color='black')
        
    for k in range(0,box_length//partition_size+1,1):
        ax.plot([0,box_width//partition_size],
                 [k,k],
                 linewidth = 2,
                 color='black')
    #
    
def plot_networkx(transition_matrix,
                 dimensionality):
    
    import networkx as nx

    T = transition_matrix.copy()
    np.fill_diagonal(T, 0)

    # 
    thresh = 1
    idx = np.where(T<thresh)
    T[idx]=0
    idx = np.where(T>=thresh)
    T[idx]=1

    #
    G = nx.from_numpy_matrix(T)
    # G = nx.DiGraph(T)
    idx = list(nx.isolates(G))
    G.remove_nodes_from(idx)
    # G.remove_nodes_from(idx)

    #pos = nx.circular_layout(G)
    pos = nx.spring_layout(G)
    #
    nx.draw(G, 
            pos, 
            node_color=range(T.shape[0]-len(idx)), 
            node_size=100, 
                cmap=plt.cm.viridis)     
    
def load_csv(fname):

    from numpy import genfromtxt
    locs = genfromtxt(fname, 
                      delimiter=',', 
                      dtype='str')

    print ("Note... saving all locations regardless of likelihood...")
    print (locs.shape)
    
    return locs
