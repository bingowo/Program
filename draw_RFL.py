import matplotlib.pyplot as plt
import tqdm
import itertools

# 模拟数据
x = []
y = []


with open('RFL.txt', 'r', encoding='utf8') as f:
    cnt = 0
    for s in tqdm.tqdm(itertools.zip_longest(*[f] * 1), desc='loading data:    ', mininterval=2):
        cnt += 1
        print(s)
        x.append(cnt)
        y.append(float(s[0]))

from scipy.signal import savgol_filter

y = savgol_filter(y, 49, 3, mode= 'nearest')

# 绘制折线图
plt.plot(x, y)

# 添加标题、轴标签和图例
#plt.title('Two Lines Plot')
plt.xlabel('data')
plt.ylabel('result')
plt.legend(loc='lower right')

# 显示图像
plt.show()
