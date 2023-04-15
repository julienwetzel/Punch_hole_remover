# Import necessary libraries
import os
import sys
import argparse
import cv2 as cv
import numpy as np
from pdf2image import convert_from_path
import tempfile
import PyPDF2
from io import BytesIO
import img2pdf

from PIL import Image, TiffTags

TAGS = TiffTags.TAGS

# Dev
import pprint

# Global vartiables
debug = False
gray_transparency = False
# Définir les options de compression disponibles pour le format TIFF
compression_options = [
    "default",  # Défini la même compression que le fichier original
    "none",
    "tiff_lzw",
    "tiff_deflate",
    "tiff_adobe_deflate",
    "tiff_packbits",
    "tiff_ccitt",
    "tiff_ccitt_t4",
    "tiff_ccitt_t6",
    "tiff_jpeg",
]


def tiff_process(image, filename, compression):
    # 1. Récupérer la taille du fichier d'origine
    original_file_size = os.path.getsize(filename)

    with Image.open(filename) as original_image:
        # Récupérer les valeurs EXIF de l'image d'origine
        orig_exif = original_image.getexif()

        # Initialiser un dictionnaire vide pour stocker les métadonnées
        metadata_original = {}
        # Parcourir les clés de l'exif d'origine
        for key in orig_exif.keys():
            # Vérifier si la clé est un tag connu
            if key in TAGS:
                tag_name = TAGS[key]
                value = orig_exif.get(key)
                # Liste de clés dont la valeur correspond à une chaîne d'un dict.
                if key in [259, 262, 284, 296, 317, 50741]:
                    ext = TAGS[key, value]
                    metadata_original[tag_name] = ext

                else:
                    metadata_original[tag_name] = value
        if debug:
            # print("\nTAGS PILLOW\n")
            # pprint.pprint(TAGS)
            print("\nEXIF FILE\n")
            pprint.pprint(metadata_original)

        # Modification du mode de couleur selon le mode de compression
        # les modes CCITT demandent une image en niveau de gris
        if compression in ["tiff_ccitt", "tiff_ccitt_t4", "tiff_ccitt_t6"]:
            # Convertir l'image en noir et blanc
            img_mode = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Convertir la matrice NumPy en objet Image de Pillow avec mode '1'
            image = Image.fromarray(np.uint8(img_mode)).convert("1")
        else:
            # Convertir l'image de BGR à RGB
            img_mode = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # Convertir la matrice NumPy en objet Image de Pillow
            image = Image.fromarray(np.uint8(img_mode))

        metadata_newimage = image.info

        # Récupérer et appliquer la résolution du fichier original
        dpi = metadata_original["XResolution"]
        metadata_newimage["dpi"] = (dpi, dpi)
        if compression == "default":
            # Utiliser la même compression que l'image originale
            metadata_newimage["compression"] = metadata_original.get("compression")
        else:
            # Utiliser la compression spécifiée
            metadata_newimage["compression"] = compression

        original_mode = original_image.mode

        if compression in ["tiff_ccitt", "tiff_ccitt_t4", "tiff_ccitt_t6"]:
            image = image.convert("1")
            metadata_newimage["PhotometricInterpretation"] = "BlackIsZero"
        elif original_mode == "1" or original_mode == "L":
            image = image.convert("L")
            metadata_newimage["PhotometricInterpretation"] = "BlackIsZero"
        elif original_mode == "LA":
            if gray_transparency:
                print("avec transparence !")
                image = image.quantize().convert("RGBA")
                metadata_newimage["PhotometricInterpretation"] = "Transparency Mask"
            else:
                image = image.convert("L")
                metadata_newimage["PhotometricInterpretation"] = "BlackIsZero"
        elif original_mode == "P":
            image = image.quantize().convert("RGB")
            metadata_newimage["PhotometricInterpretation"] = "RGB"
        elif original_mode == "PA":
            if gray_transparency:
                image = image.quantize().convert("RGBA")
                metadata_newimage["PhotometricInterpretation"] = "Transparency Mask"
            else:
                image.quantize().convert("P")
                metadata_newimage["PhotometricInterpretation"] = "RGB"
        elif original_mode == "RGB":
            # image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            image = image.convert("RGB")
            metadata_newimage["PhotometricInterpretation"] = "RGB"
        elif original_mode == "RGBA":
            # image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGBA))
            image = image.convert("RGBA")
            metadata_newimage["PhotometricInterpretation"] = "Transparency Mask"
        elif original_mode == "CMYK":
            image = image.convert("CMYK")
            metadata_newimage["PhotometricInterpretation"] = "CMYK"
        elif original_mode == "YCbCr":
            # image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2YCrCb))
            image = image.convert("YCbCr")
            metadata_newimage["PhotometricInterpretation"] = "YCbCr"
        else:
            raise ValueError(f"Mode d'image non supporté: {original_mode}")

        # Mettre à jour les métadonnées de l'image
        image.info = metadata_newimage

        # Retourner l'image modifiée
        return image

        # Enregistrer la nouvelle image avec les paramètres d'origine
        # image.save('new_image.tif', dpi=(dpi, dpi), compression='raw')


# Function to get extension filename only
def get_extension(filename):
    return os.path.splitext(filename)[1].lower()


# Function to get program filename without extension
def get_exec_name():
    exec_path = sys.argv[0]
    exec_name = os.path.basename(exec_path)
    return os.path.splitext(exec_name)[0]


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


def pdf_process(image, filename, compression):
    # Récupérer le DPI de l'image
    dpi = image.info.get("dpi", None)
    print(f"DPI de l'image : {dpi}")
    sys.exit()


def get_image_dpi(img):
    # Récupérer les dimensions de l'image en pixels
    height, width = img.shape[:2]

    # Récupérer les informations de l'image
    im = Image.fromarray(img)
    dpi = im.info.get("dpi", None)

    if dpi is None:
        # Calculer la résolution de l'image en DPI
        screen_dpi = 96  # Valeur par défaut pour les écrans Windows
        height_inches = height / screen_dpi
        width_inches = width / screen_dpi
        dpi = int(max(height, width) / max(height_inches, width_inches))

    return dpi


def get_image_ppp(img):
    # Récupérer les dimensions de l'image en pixels
    height, width = img.shape[:2]

    # Récupérer les informations de l'image
    im = Image.fromarray(img)
    dpi = im.info.get("dpi", None)

    if dpi is None:
        # Calculer la résolution de l'image en DPI
        screen_dpi = 96  # Valeur par défaut pour les écrans Windows
        height_inches = height / screen_dpi
        width_inches = width / screen_dpi
        dpi = int(max(height, width) / max(height_inches, width_inches))

    ppp = dpi * 0.03937  # Conversion de DPI à ppp

    return ppp


# Define the main function
def main(argv):
    parser = argparse.ArgumentParser(
        description="Punch Hole Remover: Removes punch holes from images or pages of PDF files."
    )
    parser.add_argument(
        "filename",
        help="Name of the image or PDF file to process (supported formats: .tiff).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "--compression",
        choices=compression_options,
        default="default",
        help="option de compression à utiliser (par défaut : default)",
    )
    parser.add_argument(
        "--graytransparency", action="store_true", help="Enable transparency channel"
    )
    args = parser.parse_args(argv)

    global gray_transparency
    # Définit si l'utilisateur veut garder ou pas le canal alpha des images
    gray_transparency = args.graytransparency is not None and args.graytransparency

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

    # In developpment
    if file_extension == ".pdf":
        # Ouvrir le fichier PDF
        with open(filename, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Itérer sur toutes les pages du PDF
            for page_num in range(len(pdf_reader.pages)):
                # Extraire la page du PDF
                page = pdf_reader.pages[page_num]
                test = pdf_reader.get_page_number(0)
                xObject = page["/Resources"]["/XObject"].get_object()

                # Chercher la première image dans la page
                for obj in xObject:
                    if xObject[obj]["/Subtype"] == "/Image":
                        # Obtenir les informations de l'image
                        width = xObject[obj]["/Width"]
                        height = xObject[obj]["/Height"]
                        data = xObject[obj]._data

                        # Charger l'image dans OpenCV
                        img = cv.imdecode(np.frombuffer(data, np.uint8), -1)
                        # img = cv.equalizeHist(img)
                        hole_remover_image(img)
                        # img_tiff_default = pdf_process(img, filename, args.compression)

                        # Récupérer le DPI de l'image
                        dpi = get_image_dpi(img)
                        print(f"DPI de l'image : {dpi}")

                        # Récupérer la résolution en ppp de l'image
                        ppp = get_image_ppp(img)
                        print(f"Résolution en ppp de l'image : {ppp}")
                        sys.exit()

                        if debug:
                            # Optionally, save each page as an image
                            result_filename = (
                                f"{get_exec_name()}__{filename}__{page_num}__result.png"
                            )
                            cv.imwrite(result_filename, img)
                        break
        """# Convert PDF pages to images
        pages = convert_from_path(filename)
        for i, page in enumerate(pages):
            # Convert the PIL image to OpenCV format
            page = np.array(page)
            page = cv.cvtColor(page, cv.COLOR_RGB2BGR)
            hauteur, largeur, canaux = page.shape
            print("La résolution de l'image est : {} x {}".format(largeur, hauteur))

            # Process the image
            # hole_remover_image(page)
            hole_remover_image(page)
            # converting into chunks using img2pdf
            # pdf_bytes = img2pdf.convert(image.filename)

            if debug:
                # Optionally, save each page as an image
                result_filename = f"{get_exec_name()}__{filename}__{i}_result.png"
                cv.imwrite(result_filename, page)
            else:
                # Replace the original file with the result
                cv.imwrite(filename, page)"""

    elif file_extension in [".tiff", ".tif"]:
        # Charger une image avec OpenCV
        image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_UNCHANGED)

        # Check if the image was loaded successfully
        if image is None:
            print("Error opening image!")
            parser.print_help()
            return -1

        # Process the image
        hole_remover_image(image)
        # image = cv.equalizeHist(image)
        # Mise en forme de l'image
        img_tiff_default = tiff_process(image, filename, args.compression)

        if debug:
            # Enregistrer l'image tiff avec les même paramètres que le fichier original
            filename = f"debug_{filename}"
            img_tiff_default.save(filename)
        else:
            # Enregistrer l'image tiff avec les même paramètres que le fichier original
            img_tiff_default.save(filename)
    else:
        sys.exit("Le format de fichier n'est actuellement pas supporté")


# Call the main function
if __name__ == "__main__":
    main(sys.argv[1:])
