import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('legend', fontsize=20)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)

# Adversarial data
llava_one_adv = np.array([
    # run scrip run_score_for_each_topk.py to having the result
])

llava_next_adv = np.array([
    # run scrip run_score_for_each_topk.py to having the result
])

qwenvl2_adv = np.array([
    # run scrip run_score_for_each_topk.py to having the result
])

deepseek_vl2_adv = np.array([
    # run scrip run_score_for_each_topk.py to having the result
])

# Random data
llava_one_rand = np.array([
    # run script run_score_for_each_topk.py to having the result
])

llava_next_rand = np.array([
    # run script run_score_for_each_topk.py to having the result
])

qwenvl2_rand = np.array([
    # run scrip run_score_for_each_topk.py to having the result

])

deepseek_vl2_rand = np.array([
    # run script run_score_for_each_topk.py to having the result
])

# Combine data for unified color scale
all_data = np.concatenate([
    llava_one_adv.flatten(), llava_next_adv.flatten(), 
    qwenvl2_adv.flatten(), deepseek_vl2_adv.flatten(),
    llava_one_rand.flatten(), llava_next_rand.flatten(), 
    qwenvl2_rand.flatten(), deepseek_vl2_rand.flatten()
])
vmin, vmax = all_data.min(), all_data.max()

models_data = {
    "LLaVA-One.": (llava_one_adv, llava_one_rand),
    "LLaVA-Next.": (llava_next_adv, llava_next_rand),
    "Qwen2.5VL": (qwenvl2_adv, qwenvl2_rand),
    "DeepseekVL2": (deepseek_vl2_adv, deepseek_vl2_rand),
}

# Create custom colormap - smaller values = darker colors
cmap = plt.cm.bwr_r

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

def add_small_rectangles(ax, data_rand, cmap, vmin, vmax):
    """Add small rectangles for random data in bottom-right corner of each cell"""
    for i in range(data_rand.shape[0]):  # Start from row 1
        for j in range(data_rand.shape[1]):
            norm_val = (data_rand[i, j] - vmin) / (vmax - vmin)
            color = cmap(norm_val)
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if data_rand[i, j] > 0.87:
                text_color = 'white'
            else:
                text_color = 'black'

            
            
            rect = Rectangle((j + 0.52, i + 1 + 0.7), 0.48, 0.3, 
                            facecolor=color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)

            ax.text(j + 0.775, i + 1 + 0.85, f'{data_rand[i, j]*100:.1f}', 
                    ha='center', va='center', fontsize=14, color=text_color)
for ax, (name, (data_adv, data_rand)) in zip(axs, models_data.items()):
    # Create main heatmap with adversarial data
    labels = np.vectorize(lambda x: f'{x*100:.1f}')(data_adv)

    sns.heatmap(
        data_adv,
        annot=labels,
        # fmt=".3f",
        fmt="",
        cmap=cmap,
        annot_kws={"size": 24},
        xticklabels=[0, 1, 2, 3, 4, 5],
        yticklabels=[0, 1, 2, 3, 4, 5],
        cbar=False,
        ax=ax,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add small rectangles for random data
    add_small_rectangles(ax, data_rand, cmap, vmin, vmax)
    
    ax.set_title(name, fontsize=22)
    ax.set_xlabel("Retrieve Top-k images")
    ax.set_ylabel("Number of Injected Images")

# Add a single colorbar for the entire figure
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=14)

# Add legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='lightgray', edgecolor='black', label='Adversarial (main)'),
#     Patch(facecolor='darkgray', edgecolor='white', label='Random (small rectangle)')
# ]
# fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
#            ncol=2, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, right=0.92)

# Uncomment to save
plt.savefig("dual_data_heatmap.pdf", dpi=300, bbox_inches='tight')
# plt.show()