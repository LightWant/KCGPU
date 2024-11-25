# -*- coding: utf-8 -*-

import csv

# 定义输入输出文件路径
input_file = "amazon0601.txt"  # 输入的 .txt 文件路径
output_file = "amazon0601.tsv"  # 输出的 .tsv 文件路径

# 打开输入文件读取数据，输出文件写入数据
with open(input_file, "r") as infile, open(output_file, "w", newline='') as outfile:
    tsv_writer = csv.writer(outfile, delimiter='\t')
    
    # 遍历每一行
    for line in infile:
        # 去除换行符并按空格分割成节点列表
        nodes = line.strip().split()
        # 写入 .tsv 文件
        tsv_writer.writerow(nodes)

print(f"转换完成，输出文件：{output_file}")

