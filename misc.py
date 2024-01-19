original_width = 3024
original_height = 4032
scale_factor = 0.5  # 50% reduction, adjust as needed

new_width = int(original_width * scale_factor)
new_height = int(original_height * scale_factor)

print(f"Original Dimensions: {original_width} x {original_height}")
print(f"New Dimensions: {new_width} x {new_height}")