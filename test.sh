# # Declare an array
# my_array=("numpy" "pytorch" "matplotlib" "torchvision" "pandas")

# # Create an empty array to store filtered elements
# filtered_array=()

# # Loop through each element of the original array
# for element in "${my_array[@]}"; do
#   # Check if the element contains the string "torch"
#   if [[ ! "$element" == *"torch"* ]]; then
#     # Add the element to the filtered array if it contains "torch"
#     filtered_array+=("$element")
#   fi
# done

# # Print the filtered array
# echo "${filtered_array[@]}"
# required_packages+=($(cat defaults.txt | grep -v "^#"))
cut -d'=' -f1 requirements.txt

echo "${required_packages[@]}"
