import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os
import math

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['test','train']
inner_names = ['angry','disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
new_inner_names=[]
e_values=[]
for i in range(19):
    e_values.append(0)
for items in inner_names:
    if items!="neutral":
        new_inner_names.append('slightly '+items)
        new_inner_names.append('regular '+items)
        new_inner_names.append('very '+items)
    else:
        new_inner_names.append(items)


df_labels=pd.DataFrame([e_values],columns=new_inner_names)
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)
    for inner_name in new_inner_names:
        os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)

# to keep count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

df = pd.read_csv('./fer2013_range.csv')
mat = np.zeros((48,48),dtype=np.uint8)
#print('Dataframe: ', df_labels.loc[0,'regular angry'])
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        directory='train'
    else:
        directory='test'
        
    intensity_dec, intensity_num= math.modf(df['emotion'][i])
    
    if intensity_dec>0.6:
            intensity='very'
    elif intensity_dec<0.3:
            intensity='slightly'
    else:
        intensity='regular'
    
    #print(df_labels.loc[0,'regular angry'])
    if int(intensity_num) == 0:   
        img.save('data/'+directory+'/'+intensity+' angry/im'+str(df_labels.loc[0,intensity+' angry'])+'.png')
        df_labels.loc[0,intensity+' angry']+=1

    elif int(intensity_num) == 1:   
        img.save('data/'+directory+'/'+intensity+' disgusted/im'+str(df_labels.loc[0,intensity+' disgusted'])+'.png')
        df_labels.loc[0,intensity+' disgusted']+=1
    elif int(intensity_num) == 2:   
        img.save('data/'+directory+'/'+intensity+' fearful/im'+str(df_labels.loc[0,intensity+' fearful'])+'.png')
        df_labels.loc[0,intensity+' fearful']+=1
    elif int(intensity_num) == 3:   
        img.save('data/'+directory+'/'+intensity+' happy/im'+str(df_labels.loc[0,intensity+' happy'])+'.png')
        df_labels.loc[0,intensity+' happy']+=1
    elif int(intensity_num) == 4:   
        img.save('data/'+directory+'/'+intensity+' sad/im'+str(df_labels.loc[0,intensity+' sad'])+'.png')
        df_labels.loc[0,intensity+' sad']+=1
    elif int(intensity_num) == 5:   
        img.save('data/'+directory+'/'+intensity+' surprised/im'+str(df_labels.loc[0,intensity+' surprised'])+'.png')
        df_labels.loc[0,intensity+' surprised']+=1
    elif int(intensity_num) == 6:
        img.save('data/'+directory+'/neutral/im'+str(df_labels.loc[0,'neutral'])+'.png')
        df_labels.loc[0,'neutral']+= 1


print("Done!")