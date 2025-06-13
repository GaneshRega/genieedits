import sys
from rembg import remove
from PIL import Image

"""
remove_bg.py
A script to remove the background from an image using rembg.
Usage:
    python remove_bg.py input.jpg output.png
"""

def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_bg.py <input_image> <output_image>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    try:
        with Image.open(input_path) as img:
            output = remove(img)
            output.save(output_path)
        print(f"Background removed: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
