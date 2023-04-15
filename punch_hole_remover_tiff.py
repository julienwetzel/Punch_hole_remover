# Import necessary libraries
import os
import sys
import argparse
import cv2 as cv
import numpy as np

from PIL import Image, TiffTags, TiffImagePlugin

TAGS = TiffTags.TAGS

# Global vartiables
debug = False


# Function to get extension filename only
def get_extension(filename):
    return os.path.splitext(filename)[1].lower()


def hole_remover_image(image):
    # Preprocess the image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv.equalizeHist(gray)
    blur = cv.GaussianBlur(hist, (31, 31), cv.BORDER_DEFAULT)
    height, width = blur.shape[:2]

    # Define the search area
    left_margin_start_index = 0  # 0
    left_margin_end_index = width // 10  # 10
    right_margin_start_index = (width * 9) // 10  # 9 / 10
    right_margin_end_index = width
    middle_margin_start_index = (height // 2) - (height // 4)  # 2 / 4
    middle_margin_end_index = (height // 2) + (height // 4)  # 2 / 4

    # Create an empty image with the same dimensions as the original image
    search_area = np.zeros_like(blur)

    # Copy the left and right search areas from the original image into the new image
    search_area[
        middle_margin_start_index:middle_margin_end_index,
        left_margin_start_index:left_margin_end_index,
    ] = blur[
        middle_margin_start_index:middle_margin_end_index,
        left_margin_start_index:left_margin_end_index,
    ]
    search_area[
        middle_margin_start_index:middle_margin_end_index,
        right_margin_start_index:right_margin_end_index,
    ] = blur[
        middle_margin_start_index:middle_margin_end_index,
        right_margin_start_index:right_margin_end_index,
    ]

    # Set Hough Circle parameters
    minR = round(width / 80)  # 90
    maxR = round(width / 60)  # 60
    minDis = round(width / 4)  # 4

    # Detect circles
    circles = cv.HoughCircles(
        search_area,
        cv.HOUGH_GRADIENT,
        1,
        minDis,
        param1=100,  # 240
        param2=20,  # 20
        minRadius=minR,
        maxRadius=maxR,
    )

    # Fill the detected circle locations (Development)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            if debug:
                cv.circle(image, (x, y), int(r * 0.2) + r, (0, 0, 255), -1)
                cv.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            else:
                cv.circle(image, (x, y), int(r * 0.2) + r, (255, 255, 255), -1)

    # Check if the number of detected circles is not equal to 2
    circle_count = len(circles) if circles is not None and circles.any() else 0

    if debug is not True:
        if circle_count not in (0, 2):
            message = f"""Error: This program is designed to detect two punch holes, 
                        but {circle_count} punch holes were detected. 
                        Please check your input image and try again."""
            sys.exit(message)
    else:
        print(f"Detect number of holes : {circle_count}")


# Define the main function
def main(argv):
    parser = argparse.ArgumentParser(
        description="Punch Hole Remover: Removes punch holes from images"
    )
    parser.add_argument(
        "filename",
        help="Name of the image to process (supported formats: .tiff).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    args = parser.parse_args(argv)

    if args.debug:
        message = """Warning: You are about to run the program in debug mode.
        \rIn this mode, some files are created. Run this mode in a folder that is not automatically processed by another program."""
        response = input(f"{message}\nDo you want to continue? (y/n): ")
        if response.lower() != "y":
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Running the program in debug mode.")

        # Set global variable debug
        global debug
        debug = True

    # Get filename in argument
    filename = args.filename

    # Load the image
    file_extension = get_extension(filename)

    if file_extension in [".tiff", ".tif"]:
        # Charger une image avec OpenCV
        image = cv.imread(filename, cv.IMREAD_UNCHANGED)
        # Check if the image was loaded successfully
        if image is None:
            print("Error opening image!")
            parser.print_help()
            return -1

        # Supprimer les trous dans l'image
        hole_remover_image(image)

        # Récupère les informations de résolution du fichier original
        with Image.open(filename) as original_image:
            orig_exif = original_image.getexif()
            XResolutionKey, YResolutionKey, ResolutionUnitKey = 282, 283, 296
            XR, YR, RU = (
                orig_exif.get(XResolutionKey),
                orig_exif.get(YResolutionKey),
                orig_exif.get(ResolutionUnitKey),
            )

        # Insère les informations de résolution dans le fichier crée
        params = [
            cv.IMWRITE_TIFF_XDPI,
            int(XR),
            cv.IMWRITE_TIFF_YDPI,
            int(YR),
            cv.IMWRITE_TIFF_RESUNIT,
            int(RU),
        ]

        if debug:
            # Affiche les informations de résolution dans la console
            print("X Resolution:", params[1], "pixels per unit")
            print("Y Resolution:", params[3], "pixels per unit")
            print(
                "Resolution Unit:",
                params[5],
                "({})".format("inch" if params[5] == 2 else "centimeter"),
            )
            # Enregistrer l'image tiff avec les même paramètres que le fichier original
            filename = f"debug_{filename}"

            cv.imwrite(filename, image, params)
        else:
            # Enregistrer l'image tiff avec les même paramètres que le fichier original
            cv.imwrite(filename, image, params)
    else:
        sys.exit("Le format de fichier n'est actuellement pas supporté")


# Call the main function
if __name__ == "__main__":
    main(sys.argv[1:])
