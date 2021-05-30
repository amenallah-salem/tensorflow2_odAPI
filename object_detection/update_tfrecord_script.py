import os
import yaml 


def _red_config(yaml_dir):
  with open(os.path.join(yaml_dir, "cfg.yaml"), "r") as yamlfile:
    data = yaml.load(yamlfile, Loader=yaml.FullLoader)
  return data 

if __name__=='__main__':
  main_dir = os.getcwd()# inside main dir obje_det
  config_dir = '../'
  data = _red_config(config_dir)
  # class_dic = data['names']
  classes = []
  num = []
  for key, val in data['names'].items():
    classes.append(key)
    num.append(val)
  
  if len(classes) == len(num):
      beguin_line = 28
      for idx in range(0, len(classes)):
          

          print("Checking len(classes) = len(label_enc)....")
          with open('generate_tfrecord.py', 'r') as f:
              lines = f.read().split('\n')
              val = str(lines[idx + beguin_line].split(' == ')[-1])
              print('beg line',beguin_line)
              print('idx',idx)
              print('val==',val)
              enc = int(lines[idx + beguin_line + 1].split(" ")[-1])
              print('old class {} asseigned with class {} :'.format(val,enc))
              print("Updating class encoding to new class with ../cfg.yaml")

              new_line = "    if row_label == '{}':".format(classes[idx])
              new_return =  "        return {}".format(num[idx])
              print('udated new line')
              print('updating new return')
              
              lines[idx + beguin_line] = new_line # print(new_file)
              lines[idx + beguin_line +1] = new_return
              print("here erro")
              print(lines)
              beguin_line = beguin_line + 1


          with open('generate_tfrecord.py', 'w') as f:
              f.write('\n'.join(lines[:]))
              print("Finish updating and writing to tf_records.py setup for classe {}:{}".format(classes[idx], num[idx]))
          


