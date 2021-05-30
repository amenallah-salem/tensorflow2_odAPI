def _red_config(yaml_dir):
  with open(os.path.join(yaml_dir, "cfg.yaml"), "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
  return data 



# class_dic = data['names']

def label_map(num_classes):
  with open('./train_utils/label_map.pbtxt', 'a') as the_file:
    s = ' '
    for class_id in range(num_classes):
      #print(num_classes)
      the_file.write('item {\n ')
      the_file.write(s + 'id : {}'.format(int(class_id +1 )))
      the_file.write('\n')
      the_file.write(2*s +"name :'{0}'".format(class_names[class_id]))
      the_file.write('\n')
      the_file.write('}\n')


if __name__=='__main__':
  import yaml
  import os
  main_dir = os.getcwd()# inside main dir obje_det
  print("main dir now is ",main_dir)
  data = _red_config('../')
  class_names=[]
  for key, val in data['names'].items():
    class_names.append(key)
    num_classes = len(class_names)


  label_map(num_classes)
  print("label_map.pbtxt file has been created succesfuly")
