def extract_frames(input_file, output_file, start_frame, end_frame, track_ids):
    """
    从跟踪结果文件中提取指定帧范围的数据
    参数:
        input_file: 输入文件名
        output_file: 输出文件名
        start_frame: 起始帧号(包含)
        end_frame: 结束帧号(包含)
    """
    extracted_lines = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 去除行尾换行符并分割数据
            parts = line.strip().split(',')

            # 确保有足够的列且第一列是帧号
            if len(parts) >= 1 and parts[0].isdigit():
                frame_id = int(parts[0])
                track_id = int(parts[1])

                # 检查帧号是否在指定范围内
                if start_frame <= frame_id <= end_frame:
                    # adjusted_frame_id = frame_id - start_frame + 1
                    adjusted_frame_id = frame_id
                    if track_ids is not None and track_id in track_ids:
                        parts[0] = str(frame_id)
                        parts[1] = str(track_id)
                        adjusted_line = ','.join(parts) + '\n'
                        outfile.write(adjusted_line)  # 保留原始格式
                        extracted_lines += 1
                    elif track_ids is None or len(track_ids) == 0:
                        parts[0] = str(adjusted_frame_id)
                        adjusted_line = ','.join(parts) + '\n'
                        outfile.write(adjusted_line)  # 保留原始格式
                        extracted_lines += 1

    print(f"提取完成! 共提取 {extracted_lines} 行数据")
    print(f"结果已保存到: {output_file}")


# 使用示例
if __name__ == "__main__":
    input_txt = "/home/jia/PycharmProjects/gta-link/test_data/DeepEIoU_Results/1212.txt"  # 输入文件名
    output_txt = "filtered_results.txt"  # 输出文件名
    start_frame = 0  # 起始帧号
    end_frame = 2700  # 结束帧号

    extract_frames(input_txt, output_txt, start_frame, end_frame, [])
