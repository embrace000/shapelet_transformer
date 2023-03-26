import re

# 定义正则表达式模式，用于匹配函数调用行
#regex_pattern = r'<(.+):\s+(.+)\((.*)\)> -> <(.+):\s+(.+)\((.*)\)>'
regex_pattern = r'<(.+):(.+)\s(.+)\((.*)\)> -> <(.+):(.+)\s(.+)\((.*)\)>'

# 打开输入和输出文件
input_file = open('test.logcat', 'r')
output_file = open('output.log', 'w')

# 逐行读取输入文件并处理每一行
for line in input_file:
    # 使用正则表达式模式匹配行
    match = re.match(regex_pattern, line.strip())

    # 如果匹配成功，提取函数调用的来源和目标
    if match:
        source_class = match.group(1)
        source_method_type = match.group(2)
        source_method = match.group(3)
        source_args = match.group(4)
        target_class = match.group(5)
        target_method_type=match.group(6)
        target_method = match.group(7)
        target_args = match.group(8)

        #将提取的信息输出到新文件中
        output_file.write(
            f'{source_class}.{source_method}({source_args}) -> {target_class}.{target_method}({target_args})\n')

#关闭输入和输出文件
input_file.close()
output_file.close()