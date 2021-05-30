import yaml
from pathlib import Path
import os

def update_manually_config_file(main_dir):
  # with open(os.path.join(main_cfg_file, "cfg.yaml"), "r") as yamlfile:
  #   data = yaml.load(yamlfile, Loader=yaml.FullLoader)
  #   print(data)
  #   print("Reading yaml file successful")
  #   data['num_classes'] = '3'
  #   print("Value of num_classes updated from '0' to '3'")
  #   print(data)

  #   yamlfile.close()
   with open(os.path.join(main_dir, "cfg.yaml"), "r") as yamlfile:
     data = yaml.load(yamlfile, Loader=yaml.FullLoader)
     print(data)
     print("Reading yaml file successful")
     data['train_input_reader']['label_map_path'] = 'PATH_TO_BE_CONFIGURED_modified/label_map.txt'
     print("Value of num_classes updated from '0' to '3'")
     print(data)

     yamlfile.close()

def update_AUTO_config_file(main_dir, key_old_one, Value_old_one, new_one):
  with open(os.path.join(main_dir,"cfg.yaml")):
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
     print(data)
     print("Reading yaml file successful")
     data[key_old_one][Value_old_one] = new_one
     
     yamlfile.close()
# def test_yaml():
#   if 
#   else:
#     print("Something went wrong... check ../yaml.config file ")



if __name__=="__main__":
  print("checking yaml dir:")
  yaml_config_dir_dir = str(Path.cwd().parent)
  print("yaml dir")
  print(yaml_config_dir_dir)
  update_manually_config_file(yaml_config_dir_dir)
  #test_yaml()
