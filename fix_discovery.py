import re

with open('src/aegis_lab/hardware/discovery.py', 'r') as f:
    content = f.read()

# Fix the broken condition
content = re.sub(
    r'if not any\(p in precisions for p in \["INT8", "BF16", "FP16"\]\) and\n\s+\(cpu_features\["amx"\] or cpu_features\["avx_vnni"\]\):',
    'if not any(p in precisions for p in ["INT8", "BF16", "FP16"]) and (cpu_features["amx"] or cpu_features["avx_vnni"]):',
    content
)

# Remove the broken print statements
content = re.sub(r'print\("--- AEGIS-LAB Hardware SITREP ---\"\)', '', content)

with open('src/aegis_lab/hardware/discovery.py', 'w') as f:
    f.write(content)
