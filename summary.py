import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from utils import *





def setup_chinese_font():
    """配置中文字体支持"""
    try:
        # 尝试使用系统中存在的支持中文的字体
        font_names = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'STHeiti',
                      'WenQuanYi Zen Hei', 'DejaVu Sans']

        for font_name in font_names:
            if font_name in mpl.font_manager.get_font_names():
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                return True
    except:
        pass

    # 如果找不到系统字体，尝试使用默认字体并显示警告
    print("警告: 未找到合适的中文字体，中文可能显示为方框")
    return False



def plot_id_distribution(id_count, output_dir='.'):
    """绘制ID分布统计图并保存到文件"""
    if not id_count:
        print("没有找到ID数据")
        return

    # 设置支持中文的字体
    setup_chinese_font()

    # 准备数据
    ids = sorted(id_count.keys())
    counts = [id_count[i] for i in ids]

    # 创建图表
    fig = plt.figure(figsize=(12, 6))

    # 条形图
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(ids, counts, color='skyblue')
    ax1.set_title('每个ID的出现次数')
    ax1.set_xlabel('ID')
    ax1.set_ylabel('出现次数')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形上方添加数值
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # 饼图
    ax2 = fig.add_subplot(1, 2, 2)
    sizes = [c / sum(counts) * 100 for c in counts]
    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=[str(i) for i in ids],
        autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
        startangle=90,
        colors=plt.cm.tab20.colors,
        textprops={'fontsize': 8}
    )

    # 添加图例
    ax2.legend(
        wedges,
        [f'ID {i} ({c}次)' for i, c in zip(ids, counts)],
        title="ID分布",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    ax2.set_title('ID分布比例')
    ax2.axis('equal')  # 确保饼图是圆形

    plt.tight_layout()

    # 保存图表到文件
    output_path = os.path.join(output_dir, 'id_distribution.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，避免内存泄漏

    print(f"已保存ID分布图到: {output_path}")

    # 返回图像对象用于显示
    return mpimg.imread(output_path)


def main():
    # 配置路径
    data_file = "/home/jia/PycharmProjects/gta-link/test_data/DeepEIoU_Results/1212.txt"  # MOT数据文件路径

    # 读取数据
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # 解析数据
    frame_data, id_count = parse_mot_data(data_lines)


    # 统计信息
    total_ids = len(id_count)
    total_boxes = sum(id_count.values())
    print(f"统计结果:")
    print(f"总ID数量: {total_ids}")
    print(f"总检测框数量: {total_boxes}")
    print("\n每个ID的检测框数量:")
    for obj_id, count in sorted(id_count.items()):
        print(f"ID {obj_id}: {count} 个检测框")


    # 绘制统计图
    # plot_id_distribution(id_count)


if __name__ == "__main__":
    main()
