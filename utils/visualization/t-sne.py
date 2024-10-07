#from openTSNE import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import sys
import pandas as pd

def visualize(
    x,
    y,
    save_path,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=True,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    colors = {'angry_level1': '#d62728', 'angry_level2': '#d62728', 'calm_level1': '#bcbd22', 'calm_level2': '#bcbd22',
 'disgust_level1': '#9467bd', 'disgust_level2': '#9467bd', 'fearful_level1': '#e377c2', 'fearful_level2': '#e377c2',
 'happy_level1': '#ff7f0e', 'happy_level2': '#ff7f0e', 'neutral_level1': '#1f77b4', 'sad_level1': '#8c564b', 'sad_level2': '#8c564b',
 'surprised_level1': '#2ca02c', 'surprised_level2': '#2ca02c'} 
    '''{'#9467bd'紫, '#ff7f0e'橙, 
    '#bcbd22'黄, 
    '#d62728'红, '#1f77b4'蓝, '#17becf'青, '#8c564b'棕, '#e377c2'粉, '#2ca02c'绿}'''
    if ax is None:
        fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 4)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = (y == yi)
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)
        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 10),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="best", bbox_to_anchor=(0.05, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
    
    fig.savefig(save_path)

'''tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)'''

tsne = TSNE(perplexity=30,n_components=2, init='pca', n_iter=2000)

emotion_dic = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'}
# 读取npz文件
audio_dic=np.load('/data/home/yuqihang/TallkingFace/EmoGau/audio_data_ravdess.npz')
x_audio=np.array(audio_dic['embedding'])
label_audio=np.array(audio_dic['label'])
level_audio=np.array(audio_dic['level'])

y_audio = np.array([f'{emotion_dic[int(c1)]}_level{int(c2)}' for c1, c2 in zip(label_audio, level_audio)])
#y_audio = np.array([f'{c1}_{c2}' for c1, c2 in zip(label_audio, level_audio)])

t_sne = tsne.fit_transform(x_audio)

x_coords = t_sne[:,0]
y_coords = t_sne[:,1]
df = pd.DataFrame({  
    'x': x_coords,  
    'y': y_coords,  
    'group': y_audio  
}) 

visualize(t_sne,y_audio,'audio_ravdess.png')
sys.exit(0)
type_audio=np.zeros(y_audio.shape[0])

hdtf_dic=np.load('/data/home/yuqihang/TallkingFace/EmoGau/AVCC_result/0.5/audio_data_hdtf.npz')
x_hdtf=np.array(hdtf_dic['embedding'])
y_hdtf = np.array(['hdtf' for i in range(x_hdtf.shape[0])])
type_hdtf=np.ones(y_hdtf.shape[0])

x=np.concatenate((x_audio, x_hdtf))
label=np.concatenate((y_audio, y_hdtf))
tp=np.concatenate((type_audio, type_hdtf))

t_sne = tsne.fit_transform(x) # array(float64) [B,50]==>[B, 2]

x_audio = t_sne[tp == 0]
x_hdtf = t_sne[tp == 1]
visualize(t_sne,tp,'ravdess_hdtf.png')
#visualize(x_audio,y_audio,'audio_hdtf.png')
#visualize(x_hdtf,y_hdtf,'hdtf.png')

result={}
result['t-sne']=t_sne
result['label']=label
result['tp'] = tp
np.savez('t-sne_hdtf_reault.npz', **result)

'''
video_dic=np.load('/data/home/yuqihang/TallkingFace/EmoGau/AVCC_result/0.25/video_data_ravdess.npz')
x_video=np.array(video_dic['embedding'])
label_video=np.array(video_dic['label'])
level_video=np.array(video_dic['level'])

y_video = np.array([f'{emotion_dic[int(c1)]}_level{int(c2)}' for c1, c2 in zip(label_video, level_video)])
#t_sne = tsne.fit_transform(x_video)
#visualize(t_sne,y_video,'video.png')
type_video=np.ones(y_video.shape[0])

x=np.concatenate((x_audio, x_video))
label=np.concatenate((y_audio, y_video))
tp=np.concatenate((type_audio, type_video))

t_sne = tsne.fit_transform(x) # array(float64) [B,50]==>[B, 2]

x_audio = t_sne[tp == 0]
x_video = t_sne[tp == 1]
visualize(x_audio,y_video,'audio.png')
visualize(x_video,y_video,'video.png')

result={}
result['t-sne']=t_sne
result['label']=label
result['tp'] = tp
np.savez('t-sne_reault.npz', **result)
'''