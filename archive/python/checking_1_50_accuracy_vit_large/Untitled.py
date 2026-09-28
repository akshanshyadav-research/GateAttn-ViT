#!/usr/bin/env python3
# Original notebook: checking_1_50_accuracy_vit_large/Untitled.ipynb
# Archival export. See docs/ARCHIVE.md before executing.
if __name__ != "__main__":
    raise ImportError("Archival experiment: use the gateattn package for imports.")

# %% Cell 0 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks1"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(0, 8):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 1 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks1"
output_folder = "output1"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 2 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks2"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(8, 16):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 3 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks2"
output_folder = "output2"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 4 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks3"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(16, 24):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 5 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks3"
output_folder = "output3"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 6 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks4"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(24, 32):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 7 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks4"
output_folder = "output4"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 8 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks5"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(32, 40):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 9 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks5"
output_folder = "output5"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 10 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks6"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(40, 48):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 11 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks6"
output_folder = "output6"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

# %% Cell 12 (code)
import nbformat
import os

# Path to your original notebook
original_notebook = "version_40_combine_0nly_attention_and_similarity.ipynb"
output_folder = "notebooks7"
os.makedirs(output_folder, exist_ok=True)

# Load the original notebook
with open(original_notebook) as f:
    nb = nbformat.read(f, as_version=4)

# Generate 51 copies with drop_patch_percentage_entropy from 0 to 50
for i in range(48, 54):
    entropy_value = i  # 0 to 50
    new_nb = nb.copy()

    if new_nb['cells'][0]['cell_type'] == 'code':
        modified_lines = []
        for line in new_nb['cells'][0]['source'].splitlines():
            if line.strip().startswith("drop_patch_percentage_entropy="):
                modified_lines.append(f"drop_patch_percentage_entropy={entropy_value}")
            else:
                modified_lines.append(line)
        new_nb['cells'][0]['source'] = "\n".join(modified_lines)

    # Save the modified notebook
    new_notebook = f"{output_folder}/notebook_entropy_{entropy_value:02d}.ipynb"
    with open(new_notebook, "w") as f:
        nbformat.write(new_nb, f)

    print(f"Created {new_notebook} with drop_patch_percentage_entropy={entropy_value}")


# %% Cell 13 (code)
import os
from concurrent.futures import ThreadPoolExecutor
import papermill as pm

# Folder containing the notebooks
notebook_folder = "notebooks7"
output_folder = "output7"
os.makedirs(output_folder, exist_ok=True)

# List all notebook files
notebooks = [f"{notebook_folder}/{nb}" for nb in os.listdir(notebook_folder) if nb.endswith(".ipynb")]

# Function to run each notebook
def run_notebook(notebook):
    output_nb = f"{output_folder}/{os.path.basename(notebook)}"
    try:
        pm.execute_notebook(
            notebook,
            output_nb
        )
        print(f"Executed {notebook}, output saved to {output_nb}")
    except Exception as e:
        print(f"Failed: {notebook} with error {e}")

# Parallel execution
num_workers = min(300, os.cpu_count())  # Use available CPU cores or 100 threads
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(run_notebook, notebooks)

