import os 
import glob
import json

annot_list = glob.glob(os.path.join('./80_annotations/','*.json'))

for path in annot_list:
    with open(path , 'r' ,encoding = 'utf-8') as f:
        jf = json.load(f)
    for data in jf['images']:
        data['file_name'] = data['file_name'].split('\\')[-1]
    with open(path , 'w' ,encoding = 'utf-8') as f:
        json.dump(jf , f)