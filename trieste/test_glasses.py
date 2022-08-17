import numpy as np
import tensorflow as tf
import time

import gpflow
import tensorflow_probability as tfp
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.gpflow import build_gpr
from trieste.objectives import (
    scaled_branin,
    SCALED_BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
)
from trieste.experimental.plotting import plot_function_plotly
from trieste.space import Box
import trieste
from trieste.acquisition import LocalPenalization
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition import (
    SingleModelAcquisitionBuilder,
    ExpectedImprovement,
    Product,
)
from trieste.acquisition.function.non_myopic import Glasses, EfficientGlobalOptimization_qEI
from trieste.acquisition import (
    SingleModelAcquisitionBuilder,
    ExpectedImprovement,
    Product,
)
from utils import EIpu
from trieste.utils import to_numpy

from trieste.acquisition.optimizer import generate_random_search_optimizer

from trieste.acquisition.function import BatchMonteCarloExpectedImprovement

np.random.seed(1793)
tf.random.set_seed(1793)

acquisition = 'Glasses'
num_steps = 30

search_space = BRANIN_SEARCH_SPACE  # predefined search space, for convenience
search_space = Box([0, 0], [1, 1])  # define the search space directly
observer = trieste.objectives.utils.mk_observer(scaled_branin)

num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
model = GaussianProcessRegression(gpflow_model, num_kernel_samples=100)


if acquisition == 'EI':
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(num_steps, initial_data, model)


elif acquisition == 'EIpu':
    class local_scale(SingleModelAcquisitionBuilder):
        def prepare_acquisition_function(self, model, dataset=None):

            def acquisition(at): # [N, 1, d] -> [N,1]
                x_old = dataset.query_points[-1,:]
                return tf.exp(-tf.math.reduce_sum((at -x_old)**2,axis=-1)/(2*0.01))
            return acquisition

    ei = ExpectedImprovement().using('OBJECTIVE')
    scale = local_scale().using('OBJECTIVE')
    acq_fn = Product(ei, scale)
    rule = EfficientGlobalOptimization(acq_fn)  # type: ignore

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(
        num_steps, initial_data, model, acquisition_rule=rule
    )

elif acquisition == 'Glasses':
    lookahead = 1
    acq_fn = Glasses(search_space, lookahead)
    rule = EfficientGlobalOptimization(acq_fn, optimizer=generate_random_search_optimizer(500))
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)
    acquisition += f' {lookahead}'


elif acquisition == 'Binoculars':
    monte_carlo_sample_size = 1000
    batch_ei_acq = BatchMonteCarloExpectedImprovement(
        sample_size=monte_carlo_sample_size, jitter=1e-5
    )
    batch_size = 5
    batch_ei_acq_rule = EfficientGlobalOptimization_qEI(  # type: ignore
        num_query_points=batch_size, builder=batch_ei_acq
    )
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(
        num_steps, initial_data, model, acquisition_rule=batch_ei_acq_rule
    )
    acquisition += f' {batch_size}'


dataset = result.try_get_final_dataset()

query_point, observation, arg_min_idx = result.try_get_optimal_point()

print(f"query point: {query_point}")
print(f"observation: {observation}")


query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
global_min = SCALED_BRANIN_MINIMUM


## plotting GIFs

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

steps = np.arange(1,num_steps+1,1)
y = observations[num_initial_points:,0]
regret = []
for i in range(num_steps):
    regret_val = np.min(y[:i+1]) - to_numpy(global_min)[0]
    print(np.min(y[:i+1]), to_numpy(global_min)[0])
    print(regret_val)
    regret.append(np.log(regret_val+1e-6))

x1 = query_points[num_initial_points:,0]
x2 = query_points[num_initial_points:,1]
L1_dist = [0]
for i in range(num_steps-1):
    L1_dist.append(L1_dist[-1] + abs(x1[i+1]-x1[i]) + abs(x2[i+1]-x2[i]))

fig = plt.figure(figsize=(12,3.2))
bottom = 0.2
top = 0.85
left = 0.1
right = 0.9
gs = GridSpec(1, 3, figure=fig,
            wspace=0.5, hspace=0.2, left=left, right=right, bottom=bottom, top=top
            )
ax_list = [fig.add_subplot(gs[0, i]) for i in range(3)]

ax_list[0].set_xlim([-0,1])
ax_list[0].set_ylim([-0,1])
ax_list[0].set_xticks([0,1])
ax_list[0].set_yticks([0,1])
ax_list[1].set_xlim([0,num_steps])
ax_list[2].set_xlim([0,num_steps])

ax_list[1].set_ylim([-20,0])
ax_list[2].set_ylim([0,20])

ax_list[0].set_xlabel('x1')
ax_list[0].set_ylabel('x2')
ax_list[0].set_title('sample locations')

ax_list[1].set_xlabel('step')
ax_list[1].set_ylabel('log regret')
ax_list[1].set_title(f'{acquisition}')

ax_list[2].set_xlabel('step')
ax_list[2].set_ylabel('cost')
ax_list[2].set_title('L1 accumulated cost')


def contour(ax):
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    levels = [-1,0,1,2,3,4]

    xmin, xmax = 0,1
    ymin, ymax = 0,1

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = to_numpy(scaled_branin(positions.T)).reshape(xx.shape)
    
    # cmap = plt.cm.get_cmap('RdBu')
    cmap = plt.cm.get_cmap('Blues')
    cs = ax.contourf(xx, yy, f, levels, cmap=cmap, extend='max', origin='lower')
    ax.set_xlabel('input 1', labelpad=1.5)


    vmin, vmax = levels[0],levels[-1]
    axcolor = inset_axes(ax,
                width="5%",  # width = 5% of parent_bbox width
                height="100%",  
                loc='lower left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
                )
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    col_map = plt.get_cmap('Blues')
    cbar = mpl.colorbar.ColorbarBase(axcolor, cmap=col_map, orientation = 'vertical', norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar.set_ticks([levels[0],levels[-1]])

contour(ax_list[0])

def animate(i):
    ax_list[0].scatter(x1[i],x2[i], marker='.', color='salmon',s=100)
    ax_list[1].plot(regret[:i+1],  color='salmon')
    ax_list[2].plot(L1_dist[:i+1], color='salmon')

ani = FuncAnimation(fig, animate, frames=num_steps, interval=50)
ani.save(f"/user/home/ad20999/BORL/plots/{acquisition}.gif", dpi=300, writer=PillowWriter(fps=5))


# ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=20)    
# ani.save("/user/home/ad20999/BORL/plots/test.gif", dpi=300, writer=PillowWriter(fps=2))

