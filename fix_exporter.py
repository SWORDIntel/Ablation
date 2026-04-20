with open('src/aegis_lab/quantization/exporter.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print(' in line and 'Testing with' in line:
        line = line.replace('"', '').replace('Testing with', 'print("Testing with') + '")\n'
    new_lines.append(line)

with open('src/aegis_lab/quantization/exporter.py', 'w') as f:
    f.writelines(new_lines)
