#!/bin/bash

# generate requirements.txt file using pipreqs
pipreqs . --force
pip install --upgrade pip

# get a list of all packages in the conda environment
conda_packages=($(conda list | tail -n +4 | awk '{print $1}'))
pip_packages=($(pip list | tail -n +3 | awk '{print $1}'))

# add a list of packages that should not be uninstalled
static_packages=($(cat defaults.txt | grep -v "^#"))
required_packages+=($(cut -d'=' -f1 requirements.txt | grep -v "^#")) # get a list of required packages from the requirement.txt file
required_packages+=($(conda list | grep "^_" | awk '{print $1}'))   #any package with underscore at the beginning

# Function to check if a package name appears in the array
package_is_static() {
  local package=$1
  for pattern in "${static_packages[@]}"; do
    # Check if package name contains the pattern (with overlapping strings)
    if [[ "$package" == *"$pattern"* ]]; then
      return 0  # Package name is in the array
    fi
  done
  return 1  # Package name is not in the array
}

echo "PIP*********************************************************************************************"
pip_removed=()
for package in "${pip_packages[@]}"; do
  #Check if the package is not one of the required packages and the static package
  if [[ ! " ${package} " =~ " ${required_packages[@]} " ]] && \
     ! package_is_static " ${package} "; then
    pip uninstall -y "${package}"
    pip_removed+=("${package}")
    # echo "${package}"
  fi
done

echo "CONDA*********************************************************************************************"
package_is_pip_removed() {
  local package=$1
  for pattern in "${pip_removed[@]}"; do
    # Check if package name contains the pattern (with overlapping strings)
    if [[ "$package" == *"$pattern"* ]]; then
      return 0  # Package name is in the array
    fi
  done
  return 1  # Package name is not in the array
}

for package in "${conda_packages[@]}"; do
  #Check if the package is not one of the required packages and the static package
  if [[ ! " ${package} " =~ " ${required_packages[@]} " ]] && \
     ! package_is_static " ${package} " && \
     ! package_is_pip_removed " ${package} "; then
    conda remove -y "${package}"
    # echo "${package}"
  fi
done
