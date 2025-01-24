import struct
from PIL import Image
import os

# Folder name
gnt_folder = "E:\\Work Folder\\0_Polya\\ДТЕУ\\Дипломна робота\\HWDB1.1tst_gnt"
output_root_dir  = "output_images"  # Directory to save the images

def process_gnt_file(filepath, output_dir):
    try:
        with open(filepath, "rb") as image_file:
            n = 0  # Counter for the number of character images
            file_position = 0
            
            # Get the total file length
            image_file.seek(0, 2)  # Move to the end
            file_length = image_file.tell()
            image_file.seek(0, 0)  # Move back to the beginning

            while file_position < file_length:  # While not EOF
                # Read sample size (4 bytes, unsigned int)
                char_length_data = image_file.read(4)
                if len(char_length_data) < 4:
                    break
                char_length = struct.unpack("<I", char_length_data)[0]  # Little-endian unsigned int

                # Read tag code (2 bytes, char)
                char_label = image_file.read(2)  # 2-byte GB code
                char_label = char_label[::-1]  # Convert to human-readable format

                # Read width (2 bytes, unsigned short)
                char_width = struct.unpack("<H", image_file.read(2))[0]

                # Read height (2 bytes, unsigned short)
                char_height = struct.unpack("<H", image_file.read(2))[0]

                # Read bitmap (width x height bytes)
                bitmap_size = char_width * char_height
                char_image = image_file.read(bitmap_size)
                
                if len(char_image) < bitmap_size:
                    print("Incomplete character image detected. Stopping.")
                    break

                # Convert the bitmap to a grayscale image and save it
                img = Image.frombytes("L", (char_width, char_height), char_image)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Adjust orientation if necessary
                
                # Save the image as a PNG file
                gb_code = char_label.hex().upper()
                output_path = os.path.join(output_dir, f"image_{n + 1:04d}_GB_{gb_code}.png")
                img.save(output_path)

                # Move to the next character sample
                n += 1
                file_position = image_file.tell()

            print(f"Processed and saved {n} character images from {filepath}")
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")

def process_all_gnt_files(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    # Iterate through all .gnt files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".gnt"):
            file_path = os.path.join(input_folder, filename)
            output_dir = os.path.join(output_folder, os.path.splitext(filename)[0])

            # Create an output subdirectory for each .gnt file
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Processing {filename}...")
            process_gnt_file(file_path, output_dir)

# Process all .gnt files in the folder
process_all_gnt_files(gnt_folder, output_root_dir)

