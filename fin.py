import os
from PIL import Image
from transformers import pipeline
# from rembg import remove
# def remove_background(input_folder, output_folder, log_func=None):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     # Process each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".png")or filename.endswith(".jpg")or filename.endswith(".jpeg")or filename.endswith(".PNG"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#             # Load the image and remove the background
#             input_img = Image.open(input_path)
#             output_img = remove(input_img)
#             # Save the output image
#             output_img.save(output_path, "PNG")
#             print(f"Removed background from {filename}")
#             log_func(f"Removed background from {filename}\n") if log_func else None
#         else:
#             print(f"Skipping {filename}: not a PNG file")
#             log_func(f"Skipping {filename}: not a PNG file\n") if log_func else None
# def remove_background(input_folder, output_folder,pipe, log_func=None):
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     # Process each file in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".png")or filename.endswith(".jpg")or filename.endswith(".jpeg")or filename.endswith(".PNG"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#             # Load the image and remove the background
#             input_img = Image.open(input_path).convert("RGB")
#             # remove the background
#             output_img = pipe(input_img)
#             # Save the output image
#             output_path = os.path.join(output_folder, f"{filename}.png")
#             output_img.save(output_path, "PNG")
#             print(f"Removed background from {filename}")
#             log_func(f"Removed background from {filename}\n") if log_func else None
#         else:
#             print(f"Skipping {filename}: not a PNG file")
#             log_func(f"Skipping {filename}: not a PNG file\n") if log_func else None
def remove_background(input_folder, output_folder, log_func=None):
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.lower() != "background.jpg":
            input_path = os.path.join(input_folder, filename)
            # Strip the original extension and append ".png"
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            # Load the image and remove the background
            input_img = Image.open(input_path).convert("RGB")
            output_img = pipe(input_img)
            # Save the output image
            output_img.save(output_path, "PNG")
            print(f"Removed background from {filename}")
            if log_func:
                log_func(f"Removed background from {filename}\n")
            # remove the original image
            os.remove(input_path)
        elif filename.lower() == "background.jpg":
            print(f"Skipping {filename}: background image")
            if log_func:
                log_func(f"Skipping {filename}: background image\n")
        else:
            print(f"Skipping {filename}: not a PNG, JPG, or JPEG file")
            if log_func:
                log_func(f"Skipping {filename}: not a PNG, JPG, or JPEG file\n")

remove_background('./in', './output_images', log_func=None)