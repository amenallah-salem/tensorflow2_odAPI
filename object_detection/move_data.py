import argparse
import shutil
import os

parser= argparse.ArgumentParser()
parser.add_argument("--src_data", 
                    type=str,
                    metavar='',
                    required=True,
                    help="the source data: Where the data is first located")
parser.add_argument("--destination_data", 
                    type=str,
                    metavar='',
                    required=True,
                    help= "destination : Where the data will be stored")

args= parser.parse_args()


def copy_images(source, destination): 
  print("moving data from the source...")
  print('Listing founded files and directories Before moving folders:',os.listdir(source))
  print(100*'-')

  dest = shutil.copytree(source, destination)  
  print("After moving file:")  
    
  # Print path of newly  
  # created file  
  print("Listing found files and directories in Destination path:{} after moving folders: {} ".format(dest, os.listdir(destination)))


if __name__=='__main__':

  args= parser.parse_args()#read arguments from command line 
  # list_files = []
  # for files in os.listdir(os.getcwd()):
  #   list_files.append(files)
  # if 'images' in list_files:
  #   print("directory images/ alwready exist... Starting moving data to specific location")
  #   pass
  # else:
  #   print("directory images doen't exist... creating ./images folder Starting moving data to specific location")
  #   os.mkdir('images')
  #   print("successfully creating images/ directory .....")
  
  IMAGES_PATH= args.destination_data 
  print('directory to store the images into the repository', IMAGES_PATH)
  original_images_path =args.src_data 
  print('images are originally storred to contect vm ', original_images_path)

  copy_images(source= original_images_path,
          destination=IMAGES_PATH)

  # copy_images(source=os.path.join(original_images_path,'test'),
  #         destination=IMAGES_PATH)
  print('Images moved succesfuly to', IMAGES_PATH)
  # except Exception as e :
  #   print("An Exception has occured: ", e )
