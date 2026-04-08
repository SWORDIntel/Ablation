import re
files = ['src/aegis_lab/hardware/discovery.py', 'src/aegis_lab/quantization/exporter.py']
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    content = re.sub(r'print\("\n', 'print("""\n', content)
    content = re.sub(r'print\("---', 'print("""---', content)
    # This is a bit naive but should fix the unterminated string literals
    with open(f, 'w') as file:
        file.write(content)
