def extract_info_from_line(line):
    parts = line.strip().split(', ')
    class_name = parts[0]
    color = parts[1]
    rgb_code = [float(x.strip("(')")) for x in parts[2:5]]
    return class_name, color, rgb_code

def get_info_from_file(file_path, line_index):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if 0 <= line_index < len(lines):
            return extract_info_from_line(lines[line_index])
        else:
            return None

file_path = 'class_name_color.txt'  # Replace with the actual file path
line_index = 0  # Replace with the desired line index (0-based)

result = get_info_from_file(file_path, line_index)

if result is not None:
    class_name, color, rgb_code = result
    print(f"Class Name: {class_name}")
    print(f"Color: {color}")
    print(f"RGB Code: {rgb_code}")
else:
    print("Invalid line index.")
