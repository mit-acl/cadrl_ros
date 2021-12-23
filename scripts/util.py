import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from network import Config

# angle_1 - angle_2
# contains direction in range [-3.14, 3.14]
def find_angle_diff(angle_1, angle_2):
    angle_diff_raw = angle_1 - angle_2
    angle_diff = (angle_diff_raw + np.pi) % (2 * np.pi) - np.pi
    return angle_diff

# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def find_nearest(array,value):
    # array is a 1D np array
    # value is an scalar or 1D np array
    tiled_value = np.tile(np.expand_dims(value,axis=0).transpose(), (1,np.shape(array)[0]))
    idx = (np.abs(array-tiled_value)).argmin(axis=1)
    return array[idx], idx

def convert_cadrl_state_to_state(cadrl_state):
    # Convert the legacy cadrl format into this repo's state representation format 
    # for the host agent
    number_examples, cadrl_state_size = np.shape(cadrl_state)
    # print cadrl_state[0:5,:]

    if Config.MAX_NUM_OTHER_AGENTS_OBSERVED in [3,4]:
        # From CADRL README.txt
        # agent_centric_state = 1 x 7 + n x 8 vector
        #    [dist_to_goal, pref_speed, cur_speed, cur_heading, vx, vy, self_radius, \
        #     other_vx, other_vy, rel_pos_x, rel_pos_y, other_radius, self_radius+other_radius, dist_2_other, is_on]

        agent_state = np.zeros([number_examples, Config.FULL_STATE_LENGTH])

        agent_state[:,Config.FIRST_STATE_INDEX+0] = cadrl_state[:,0] # host agent distance to goal
        agent_state[:,Config.FIRST_STATE_INDEX+1] = cadrl_state[:,3] # host agent heading in ego frame
        agent_state[:,Config.FIRST_STATE_INDEX+2] = cadrl_state[:,1] # host agent pref_speed
        agent_state[:,Config.FIRST_STATE_INDEX+3] = cadrl_state[:,6] # host agent radius
        num_agents_on = np.zeros(number_examples)
        for i in range(Config.MAX_NUM_OTHER_AGENTS_OBSERVED):
            is_on_inds = np.where(cadrl_state[:,14+8*i] == 1.0)
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+0+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,9+8*i] # other agent px
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+1+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,10+8*i] # other agent py
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+2+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,7+8*i] # other agent vx
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+3+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,8+8*i] # other agent vy
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+4+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,11+8*i] # other agent radius
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+5+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,12+8*i] # combined radius
            agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+6+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,13+8*i] # dist_2_other
            if Config.MULTI_AGENT_ARCH in ['WEIGHT_SHARING', 'VANILLA']:
                agent_state[is_on_inds,Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+7+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i] = cadrl_state[is_on_inds,14+8*i] # is_on
            num_agents_on[is_on_inds] += 1 # keep track of how many agents are "on", for the RNN
        
        if Config.MULTI_AGENT_ARCH == 'RNN':
            agent_state[:,0] = num_agents_on


    elif Config.MAX_NUM_OTHER_AGENTS_OBSERVED == 2:
        agent_state = np.zeros([number_examples, Config.FULL_STATE_LENGTH])
        print(cadrl_state[:3,:])
        # CADRL README.txt is incorrect......
        # [dist_to_goal, pref_speed, cur_speed, cur_heading, \
        #            other_vx, other_vy, rel_pos_x, rel_pos_y, self_radius, other_radius, self_radius+other_radius, vx, vy, dist_2_other]
        agent_state[:,0] = cadrl_state[:,0] # host agent distance to goal
        agent_state[:,1] = cadrl_state[:,5] # host agent heading in ego frame
        agent_state[:,2] = cadrl_state[:,6] # host agent pref_speed
        agent_state[:,3] = cadrl_state[:,7] # host agent radius
        agent_state[:,4:9] = cadrl_state[:,9:14] # other agent px, py, vx, vy in body frame, radius
        agent_state[:,9:11] = cadrl_state[:,14:16] # combined radius, dist btwn

    else:
        print("[regression util.py] invalid number of agents")
        assert(0)

    return agent_state

def plot_current_state_ego_frame(state, figure_name=None, axes=None):
    if axes is None:
        if figure_name is None:
            fig = plt.figure(figsize=(15, 6), frameon=False)
        else:
            fig = plt.figure(figure_name,figsize=(15, 6), frameon=False)
            plt.clf()
        ax = fig.add_subplot(1, 2, 1)
    else:
        ax = axes

    plt_colors = []
    plt_colors.append([0.8500, 0.3250, 0.0980]) # red
    plt_colors.append([0.0, 0.4470, 0.7410]) # blue 
    plt_colors.append([0.4660, 0.6740, 0.1880]) # green 
    plt_colors.append([0.4940, 0.1840, 0.5560]) # purple
    plt_colors.append([0.9290, 0.6940, 0.1250]) # orange 
    plt_colors.append([0.3010, 0.7450, 0.9330]) # cyan 
    plt_colors.append([0.6350, 0.0780, 0.1840]) # chocolate 


    ###############################
    # state vector
    ##############################
    try:
        state = np.squeeze(state)

        host_dist_to_goal = state[Config.FIRST_STATE_INDEX+0]
        host_heading = state[Config.FIRST_STATE_INDEX+1]
        host_pref_speed = state[Config.FIRST_STATE_INDEX+2]
        host_radius = state[Config.FIRST_STATE_INDEX+3]

        other_pxs = []
        other_pys = []
        other_vxs = []
        other_vys = []
        other_radii = []

        if Config.MULTI_AGENT_ARCH == 'RNN':
            num_others = int(state[0])
        else:
            num_others = int(sum([state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)-1] for i in range(Config.MAX_NUM_OTHER_AGENTS_OBSERVED)]))

        for i in range(num_others):
            other_pxs.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+0+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_pys.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+1+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_vxs.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+2+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_vys.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+3+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_radii.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+4+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])

    except:
        return
    ####################################

    plt_colors_local = plt_colors
    
    ##############################################################################################
    # first subfigure
    # convert to representation that's easier to plot

    circ1 = plt.Circle((-host_dist_to_goal, 0.0), radius=host_radius, fc='w', ec=plt_colors_local[0])
    ax.add_patch(circ1)
    # goal
    ax.plot(0.0, 0.0, c=plt_colors_local[0], marker='*', markersize=20)

    wedge = ptch.Wedge([-host_dist_to_goal, 0.0], 1.0, rad2deg(host_heading - np.pi/3), rad2deg(host_heading + np.pi/3), alpha=0.1)
    ax.add_patch(wedge)
    heading = ax.plot([-host_dist_to_goal, -host_dist_to_goal + np.cos(host_heading)], [0.0, np.sin(host_heading)], 'k--')

    # Other Agent
    for i in range(len(other_pxs)): # plot all agents that are "ON"
        # circ = plt.Circle((-host_dist_to_goal + other_px, other_py), radius=0.5, fc='w', ec=plt_colors_local[i+1])
        circ = plt.Circle((-host_dist_to_goal + other_pxs[i], other_pys[i]), radius=other_radii[i], fc='w', ec=plt_colors_local[i+1])
        ax.add_patch(circ)
        # other agent's speed
        ax.arrow(-host_dist_to_goal + other_pxs[i], other_pys[i], other_vxs[i], other_vys[i], fc=plt_colors_local[i+1], \
            ec=plt_colors_local[i+1], head_width=0.05, head_length=0.1)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # plt.legend([vel_rvo, vel_SL], ['RVO', 'Selected'])

    ax.axis('equal')
    xlim = ax.get_xlim()
    new_xlim = np.array((xlim[0], xlim[1]+0.5))
    ax.set_xlim(new_xlim)
    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # plt.draw()
    # plt.pause(0.0001)
    return

def plot_snapshot(state, real_action_one_hot, real_value, possible_actions, probs, values, figure_name=None):
    if figure_name is None:
        fig = plt.figure(figsize=(15, 6), frameon=False)
    else:
        fig = plt.figure(figure_name,figsize=(15, 6), frameon=False)
        plt.clf()

    plt_colors = []
    plt_colors.append([0.8500, 0.3250, 0.0980]) # red
    plt_colors.append([0.0, 0.4470, 0.7410]) # blue 
    plt_colors.append([0.4660, 0.6740, 0.1880]) # green 
    plt_colors.append([0.4940, 0.1840, 0.5560]) # purple
    plt_colors.append([0.9290, 0.6940, 0.1250]) # orange 
    plt_colors.append([0.3010, 0.7450, 0.9330]) # cyan 
    plt_colors.append([0.6350, 0.0780, 0.1840]) # chocolate 


    ###############################
    # state vector
    ##############################
    try:
        state = np.squeeze(state)
        print(state)

        host_dist_to_goal = state[Config.FIRST_STATE_INDEX+0]
        host_heading = state[Config.FIRST_STATE_INDEX+1]
        host_pref_speed = state[Config.FIRST_STATE_INDEX+2]
        host_radius = state[Config.FIRST_STATE_INDEX+3]

        other_pxs = []
        other_pys = []
        other_vxs = []
        other_vys = []
        other_radii = []

        if Config.MULTI_AGENT_ARCH == 'RNN':
            num_others = int(state[0])
        else:
            num_others = int(sum([state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*(i+1)-1] for i in range(Config.MAX_NUM_OTHER_AGENTS_OBSERVED)]))

        for i in range(num_others):
            other_pxs.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+0+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_pys.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+1+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_vxs.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+2+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_vys.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+3+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])
            other_radii.append(state[Config.FIRST_STATE_INDEX+Config.HOST_AGENT_STATE_SIZE+4+Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH*i])

    except:
        return
    ####################################


    plt_colors_local = plt_colors
    
    ##############################################################################################
    # first subfigure
    # convert to representation that's easier to plot
    ax = fig.add_subplot(1, 2, 1)

    circ1 = plt.Circle((-host_dist_to_goal, 0.0), radius=host_radius, fc='w', ec=plt_colors_local[0])
    ax.add_patch(circ1)
    # goal
    plt.plot(0.0, 0.0, c=plt_colors_local[0], marker='*', markersize=20)

    # find and plot best action 
    selected_action_ind = np.argmax(probs)
    selected_action = possible_actions[selected_action_ind]
    x_tmp = selected_action[0] * np.cos(selected_action[1]+host_heading) 
    y_tmp = selected_action[0] * np.sin(selected_action[1]+host_heading)
    vel_SL = plt.arrow(-host_dist_to_goal, 0.0, x_tmp, y_tmp, fc='g',\
        ec='g', head_width=0.05, head_length=0.1)

    if real_action_one_hot is not None:
        real_action_ind = np.argmax(real_action_one_hot)
        real_action = possible_actions[real_action_ind,:]
        x_SL = real_action[0] * np.cos(real_action[1]+host_heading) 
        y_SL = real_action[0] * np.sin(real_action[1]+host_heading)
        vel_rvo = plt.arrow(-host_dist_to_goal, 0.0, x_SL, y_SL, fc='y',\
            ec='y', head_width=0.05, head_length=0.1)

    wedge = ptch.Wedge([-host_dist_to_goal, 0.0], 1.0, rad2deg(host_heading - np.pi/3), rad2deg(host_heading + np.pi/3), alpha=0.1)
    ax.add_patch(wedge)
    heading = plt.plot([-host_dist_to_goal, -host_dist_to_goal + np.cos(host_heading)], [0.0, np.sin(host_heading)], 'k--')


    # Other Agent
    for i in range(len(other_pxs)): # plot all agents that are "ON"
        # circ = plt.Circle((-host_dist_to_goal + other_px, other_py), radius=0.5, fc='w', ec=plt_colors_local[i+1])
        circ = plt.Circle((-host_dist_to_goal + other_pxs[i], other_pys[i]), radius=other_radii[i], fc='w', ec=plt_colors_local[i+1])
        ax.add_patch(circ)
        # other agent's speed
        plt.arrow(-host_dist_to_goal + other_pxs[i], other_pys[i], other_vxs[i], other_vys[i], fc=plt_colors_local[i+1], \
            ec=plt_colors_local[i+1], head_width=0.05, head_length=0.1)
        


    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend([vel_rvo, vel_SL], ['RVO', 'Selected'])

    ax.axis('equal')
    xlim = ax.get_xlim()
    new_xlim = np.array((xlim[0], xlim[1]+0.5))
    ax.set_xlim(new_xlim)
    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ##############################################################################################
    # second subfigure
    ax = fig.add_subplot(1, 2, 2)
    
    plot_x = host_pref_speed*possible_actions[:,0] * np.cos(possible_actions[:,1]+host_heading)
    plot_y = host_pref_speed*possible_actions[:,0] * np.sin(possible_actions[:,1]+host_heading)
    plot_z = np.squeeze(probs)

    # Add dashed line between (0,0) and max speed forward
    plt.plot([0.0, host_pref_speed*np.cos(host_heading)],[0.0, host_pref_speed*np.sin(host_heading)],'k--')
    
    ''' plot using tripcolor (2D plot) '''
    # triang = tri.Triangulation(plot_x, plot_y)
    # color_min_inds = np.where(plot_z>0)[0]
    # if len(color_min_inds) > 0:
    #     color_min = np.amin(plot_z[color_min_inds]) - 0.05
    # else:
    #     color_min = 0.0
    # color_max = max(np.amax(plot_z),0.0)
    color_min = 0.0
    color_max = 1.0
    # plt.tripcolor(plot_x, plot_y, plot_z, shading='flat', \
    #       cmap=plt.cm.rainbow, edgecolors='k',vmin=color_min, vmax=color_max)
    # plot_x = np.hstack((plot_x, plot_x, plot_x+0.05))
    # plot_y = np.hstack((plot_y, plot_y+0.05, plot_y))
    # plot_z = np.hstack((plot_z, plot_z, plot_z))
    # plt.tripcolor(plot_x, plot_y, plot_z, shading='flat', \
    #       cmap=plt.cm.rainbow, vmin=color_min, vmax=color_max)
    plt.scatter(plot_x, plot_y, marker='+', s=1000, linewidths=4, c=plot_z, cmap=plt.cm.rainbow, vmin=color_min, vmax=color_max)
    for i, txt in enumerate(plot_z):
        ax.annotate(round(txt,3), (plot_x[i], plot_y[i]))
    plt.title('True value: %.3f, NN value: %.3f' % (real_value, values[0]))
    plt.xlabel('v_x (m/s)')
    plt.ylabel('v_y (m/s)')
    cbar = plt.colorbar()
    cbar.set_ticks([color_min,(color_min+color_max)/2.0,color_max])
    cbar.ax.set_yticklabels(['%.3f'%color_min, \
                        '%.3f'%((color_min+color_max)/2.0), \
                        '%.3f'%color_max])

    # plotting style (only show axis on bottom and left)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.draw()
    plt.pause(0.0001)
    # raw_input()

def rad2deg(rad):
    return rad*180/np.pi

def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    alpha = rgba[3]
    r = max(min((1 - alpha) * 1.0 + alpha * rgba[0],1.0),0.0)
    g = max(min((1 - alpha) * 1.0 + alpha * rgba[1],1.0),0.0)
    b = max(min((1 - alpha) * 1.0 + alpha * rgba[2],1.0),0.0)
    return [r,g,b]