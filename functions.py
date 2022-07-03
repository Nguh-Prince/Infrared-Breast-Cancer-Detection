from PIL import Image
import os

import pandas as pd
import numpy as np
import re

file_extension_regex = re.compile( "(\.(\w+))$" )

get_file_dirname_and_filename = lambda filename : ( os.path.dirname( filename ), os.path.basename( filename ) )
get_file_name_and_extension = lambda filename : ( file_extension_regex.search( filename ).group(1), file_extension_regex.search( filename ).group(2), file_extension_regex.search( filename ).group(3) )

def convert_image_to_greyscale(path_to_image, path_to_save=None):
    img = Image.open(path_to_image)
    img_gray = img.convert("L")

    if not path_to_save:
        dirname = os.path.dirname(path_to_image)
        basename = os.path.basename(path_to_image)

        file_extension = file_extension_regex.search( basename )
        if file_extension:
            file_name = file_extension.group(1)
            extension = file_extension.group(2)

            path_to_save = os.path.join( dirname, file_name + "_greyed" + extension )

    img_gray.save(path_to_save)

    return ( path_to_save, img_gray )

def convert_images_in_directory_to_greyscale(path_to_images, image_extensions=[".jpg", ".jpeg"], destination_directory='temp'):
    if not destination_directory or not os.path.exists(destination_directory):
        os.mkdir(destination_directory)

    for file in os.listdir(path_to_images):
        abs_file_path = os.path.join( path_to_images, file )

        if os.path.isfile( abs_file_path ) and os.path.exists( abs_file_path ):
            dirname, filename = get_file_dirname_and_filename( abs_file_path )
            filename, extension = get_file_name_and_extension( filename )

            if extension in image_extensions:
                convert_image_to_greyscale( abs_file_path, os.path.join( destination_directory, f"{filename}_greyed{extension}" ) )

def get_image_array(path_to_image) -> np.ndarray:
    image = Image.open(path_to_image)
    
    return np.asarray( image )

def get_data(path_to_directory=r"E:\personal_projects\breast cancer detection\Imagens e Matrizes da Tese de Thiago Alves Elias da Silva", type='train') -> pd.DataFrame:
    # path_to_directory is the main directory where the data is stored
    # type can either be train or test

    data = {
        'Image': [],
        'Label': []
    }

    subdirectory = "Desenvolvimento da Metodologia" if type == 'train' else '12 Novos Casos de Testes'

    abs_path = os.path.join( path_to_directory, subdirectory )

    sick_regex = re.compile( "DOENTES", re.I )
    number_regex = re.compile ("138")
    healthy_regex = re.compile ( "SAUDAÔòá├╝VEIS", re.I )
    image_regex = re.compile( "(\.(png|jpg|jpeg))$" )
    position_regex = re.compile( ".*\d+(-(dir|esq))" )

    position_dictionary = { 'dir': 0.5, 'esq': 1 } # dir is right and esq is left

    for root, dirs, files in os.walk( abs_path ):
        for file in files:
            if image_regex.search(file):
                abs_file_path = os.path.join( abs_path, root, file )
                flag = True
                label = None

                if sick_regex.search( abs_file_path ) and not number_regex.search(abs_file_path):
                    label = 1
                elif healthy_regex.search( abs_file_path ):
                    label = 0
                else:
                    flag = False

                if flag:
                    position = position_regex.search(file)
                    if position:
                        position = position.group(2)
                        try:
                            position = position_dictionary[position]
                        except KeyError:
                            continue
                            position = 0
                    else:
                        continue
                    
                    # convert the image to an array
                    image_array = get_image_array(abs_file_path)
                    
                    try:
                        image_array = image_array[:, :, 1]
                    except IndexError as e:
                        print(f"IndexError encountered when reshaping image: {e}")
                    
                    image_array = image_array.reshape( 1, 640*480 ).astype('float32')
                        
                    image_array /= 255

                    # append the position (left or right) to the array
                    image_array = np.concatenate( (image_array, [ [position] ]), axis=1, dtype='float32' )
                    
                    data["Image"].append( image_array )
                    data["Label"].append( label )

    return pd.DataFrame(data=data)