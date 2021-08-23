#Import general libraries (needed for functions)
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def get_GSP_counts(data, x_length, data_range):
#obtain the observed counts used in the bayesian model
#corrected for accomodation pooled data from 1Q, 2Q and 3Q interleave processes
    list_bitstring = ['0','00', '000', '100'] # all valid bistrings[]
    Y_list = []    
    for i_samples in data_range:
        row_list = []
        for c_index in range(x_length) :  
            total_counts = 0
            i_data = i_samples*x_length + c_index
            for key,val in data[i_data]['counts'].items():
                if  key in list_bitstring:
                    total_counts += val
            row_list.append(total_counts)
        Y_list.append(row_list)
    return np.array(Y_list)

# GSP plot

def gsp_plot(scale, lengths, num_samples,shots, texto, title,
             y1, y1_min, y1_max, y2, y2_min, y2_max, Y1, Y2,
             first_curve = "Standard", second_curve = "Interleaved"):
            
    import matplotlib.pyplot as plt
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)
    fig, plt = plt.subplots(1, 1, figsize = [8,5])
    plt.set_yticks(np.arange(0.0,1.1,0.1))
    plt.set_ylim([0.95-scale, 1.01])
    plt.set_ylabel("P(0)", fontsize=16)
    plt.set_xlabel("Clifford Length", fontsize=16)

    plt.plot(lengths,y1,color="purple", marker="o", lw = 0.5)
    #plt.errorbar(lengths,y1,2*sy[0:m_len],
                 #color="purple", marker='o') #not used because not visible
    plt.fill_between(lengths, y1_min, y1_max,
                    alpha=.1, color = 'purple' ) 
    
    if second_curve != None:
        plt.plot(lengths,y2,color="cyan", marker='^', lw = 0.5)
        #plt.errorbar(lengths,y2,2*sy[m_len:2*m_len],
                     #color="cyan", marker='^') #not used because not visible
        plt.fill_between(lengths, y2_min, y2_max,
                        alpha=.1, color= 'cyan' )


    for i_sample in range(num_samples):
        plt.scatter(lengths, Y1[i_sample]/shots,
                   label = "data", marker="x",color="grey")
        if second_curve != None:
            plt.scatter(lengths, Y2[i_sample]/shots,
                   label = "data", marker="+",color="grey")

    legend_list = [first_curve]
    if second_curve != None :
        legend_list.append(second_curve)
    plt.legend(legend_list,
              loc = 'center right', fontsize=10)

    plt.text(0.25,0.95, texto, transform=plt.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white'))
    plt.grid()
    plt.set_title(title,
                  fontsize=14)
