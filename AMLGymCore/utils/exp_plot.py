import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.signal import savgol_filter
import os
from matplotlib.ticker import AutoLocator, FuncFormatter
import datetime as dt
def read_dat(fname):
    base_dir = r'/data/log_AML/202404/DQN/'
    dat = '/rewards.dat'
    entries = os.listdir(base_dir)
    train_folders = [entry for entry in entries if os.path.isdir(os.path.join(base_dir, entry)) and entry.startswith(fname)][0]
    with open(base_dir+train_folders+dat, 'r') as f:
        l = f.read()
        scores1 = eval(l)
        x = [i[0] for i in scores1]
        y = [i[1] for i in scores1]
    # 应用Savitzky-Golay滤波器
    window_size = 51  # 选择一个合适的窗口大小
    polyorder = 3  # 多项式的阶数
    y1_smoothed = savgol_filter(y, window_size, polyorder)
    # y1_smoothed = y
    return x, y1_smoothed


light_green = (0/255, 158/255, 115/255)
blue = (70/255, 130/255, 180/255)
violet = (0/255, 206/255, 209/255)
purple = (128/255, 0/255, 128/255)
dark_orange = (255/255, 140/255, 0/255)
medium_orange = (255/255, 165/255, 0/255)

fp = FontProperties(size=18)
plt.figure(figsize=(21, 10))
plt.xlabel('Steps', fontproperties=fp)
plt.ylabel('Reward', fontproperties=fp)
# 设置x轴刻度标签字体大小
plt.xticks(fontsize=16)
# 设置y轴刻度标签字体大小
plt.yticks(fontsize=16)

x1, y1 = read_dat('BIZ-DQN')
x2, y2= read_dat('BIZ-Double_DQN')
x3, y3 = read_dat('BIZ-DuelingDQN')
x4, y4 = read_dat('BIZ-Rainbow')

x5, y5 = read_dat('ppo([4], (False, False))')
x6, y6 = read_dat('a2c([4], (False, False))')

print('dqn',max(y1))
print('ppo',max(y5))
print('a2c',max(y6))

# x = list(range(max(len(scores1),len(scores2),len(scores3),len(scores4),len(scores5),len(scores6))))
l1, = plt.plot(x1, y1, color=light_green, label="DQN")
l2, = plt.plot(x2, y2, color=blue, linestyle='dashdot', label="Double DQN")
l3, = plt.plot(x3, y3 , color=violet, label="Dueling DQN")
l4, = plt.plot(x4, y4, color=purple, linestyle='--', label="Rainbow")

l5, = plt.plot(x5, y5, color=dark_orange, linestyle='dashdot', label="PPO")
l6, = plt.plot(x6, y6, color=medium_orange, linestyle='dotted', label="A2C")
legend = plt.legend(loc='right', fontsize=16, bbox_to_anchor=(1.135, 0.5), borderaxespad=0, frameon=False)



# def to_million(value, pos):
#     # 将值乘以1000以转换为百万单位
#     return f"{value * 2000:.0f}"
#
# formatter = FuncFormatter(to_million)
# plt.gca().xaxis.set_major_formatter(formatter)
#
#




plt.show()


plt.savefig('example_plot.svg', format='svg', dpi=300)


# 关闭当前的绘图
plt.close()
