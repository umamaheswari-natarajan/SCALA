from pathlib import Path
import re

root = Path(".")

patterns = [
    # Multiline Windows BASE_DIR definitions
    r'BASE_DIR\s*=\s*\(\s*r?"C:\\Users\\Uma\\IIIT-B\\IIITB-IBN-ORAN-WCNC"\s*r?"\\SCALA\\synthetic-dataset"\s*\)',

    # Single-line Windows synthetic dataset
    r'BASE_DIR\s*=\s*r?"C:\\Users\\Uma\\IIIT-B\\IIITB-IBN-ORAN-WCNC\\SCALA\\synthetic-dataset"',

    # Linux synthetic dataset
    r'BASE_DIR\s*=\s*"/home/iiitb/Desktop/SCALA/synthetic-dataset"',

    # Public-dataset scripts using old SCALA root
    r'BASE_DIR\s*=\s*r?"C:\\Users\\Uma\\IIIT-B\\IIITB-IBN-ORAN-WCNC\\SCALA"',
]

replacement = 'BASE_DIR = os.path.dirname(os.path.abspath(__file__))'

changed = []

for file in root.rglob("*.py"):
    if file.name == "fix_paths.py":
        continue

    text = file.read_text(encoding="utf-8")
    new_text = text

    for pattern in patterns:
        new_text = re.sub(pattern, replacement, new_text)

    if new_text != text:
        file.write_text(new_text, encoding="utf-8")
        changed.append(str(file))

print(f"Changed {len(changed)} files:")
for file in changed:
    print(file)