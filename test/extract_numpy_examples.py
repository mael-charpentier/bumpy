import os
import re
import ast

FOLDER = "./fichiers_python_1736546799"

PROGRAM_RE = re.compile(r"#\s*Programme\s+([\d.]+)")
EXERCICE_RE = re.compile(r"des exercices")

def uses_numpy(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        if "numpy" in source:
            return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "numpy":
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy":
                return True
    return False

list_script = os.listdir(FOLDER).copy()
for file_name in list_script:
    if not file_name.endswith(".py"):
        continue

    file_path = os.path.join(FOLDER, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    programs = []
    current_prog = None
    current_lines = []

    for line in lines:
        if EXERCICE_RE.search(line):
            break  # 🔥 STOP at exercises

        match = PROGRAM_RE.search(line)
        if match:
            if current_prog:
                programs.append((current_prog, "".join(current_lines)))
            current_prog = match.group(1).replace(".", "_")
            current_lines = [line]
        elif current_prog:
            current_lines.append(line)

    if current_prog:
        programs.append((current_prog, "".join(current_lines)))

    created_files = []

    for prog_number, source in programs:
        if uses_numpy(source):
            new_name = f"{file_name}.{prog_number}.py"
            new_path = os.path.join(FOLDER, new_name)

            with open(new_path, "w", encoding="utf-8") as f:
                f.write(source)

            created_files.append(new_name)

    os.remove(file_path)
