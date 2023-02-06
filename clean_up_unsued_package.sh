#!/bin/bash

# Check if the first input is a directory
if test ! -d "$1" ; then
    echo "Error: $1 is not a directory."
    exit 1
fi

# Define a list of packages to install
packages=(pip-tools pip-check-reqs)

# Iterate over the list of packages
for package in "${packages[@]}"; do
    # Check if the package is already installed
    echo "$package is already installed"
    if ! pip freeze | grep -q "$package"; then
        # If the package is not installed, install it
        pip install "$package"
    fi
done

# Check if the requirement.txt file exists
if [ ! -f "requirements.txt" ]; then
    # If the file does not exist, generate it using pip freeze
    echo "Creating requirements.txt"
    pip freeze > requirement.txt
fi


 $1 2>&1

 | while read -r line; do
    # Process each line of stderr
    package_name=$(echo "$line" | cut -d " " -f 1)
    # Check if the package exists in the requirement.txt file
    echo "$package_name"
    if grep -q "$package_name" requirements.txt; then
        # If the package exists, remove it from the requirement.txt file using sed
        sed -i "/$package_name/d" requirements.txt
        echo "$package_name removed from requirement.txt"
fi
done