import pandas as pd
from kitti_odometry import KittiEvalOdom

eval_dir = '../results/'

eval_tool = KittiEvalOdom()
kitti_gt_dir = "./dataset/kitti/gt_poses/"
nusc_gt_dir = "./dataset/nusc/gt_poses/"


gt_dir = kitti_gt_dir
df_t, df_r, df_a, df_s = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

result_dir = '../results/KITTI'

t, r, a, s = eval_tool.eval(
    gt_dir,
    result_dir,
    alignment=None
    )

df_t = pd.concat([df_t, t])
df_r = pd.concat([df_r, r])
df_a = pd.concat([df_a, a])
df_s = pd.concat([df_s, s])

nonzero_count = (df_t != 0).sum(1)
df_t['Sum'] = df_t.sum(axis=1)
mean = [df_t['Sum'][i]/nonzero_count[i] for i in range(len(nonzero_count))]
df_t['Mean'] = mean
df_t['Non-zero'] = nonzero_count

nonzero_count = (df_r != 0).sum(1)
df_r['Sum'] = df_r.sum(axis=1)
mean = [df_r['Sum'][i]/nonzero_count[i] for i in range(len(nonzero_count))]
df_r['Mean'] = mean
df_r['Non-zero'] = nonzero_count

nonzero_count = (df_a != 0).sum(1)
df_a['Sum'] = df_a.sum(axis=1)
mean = [df_a['Sum'][i]/nonzero_count[i] for i in range(len(nonzero_count))]
df_a['Mean'] = mean
df_a['Non-zero'] = nonzero_count

nonzero_count = (df_s != 0).sum(1)
df_s['Sum'] = df_s.sum(axis=1)
mean = [df_s['Sum'][i]/nonzero_count[i] for i in range(len(nonzero_count))]
df_s['Mean'] = mean
df_s['Non-zero'] = nonzero_count

print(df_t)
print(df_r)
print(df_a)
print(df_s)