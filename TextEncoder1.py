import open_clip
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
# read text file
f = open("storage.googleapis.com_bit_models_imagenet21k_wordnet_ids.txt", "r")
f1 = open("storage.googleapis.com_bit_models_imagenet21k_wordnet_lemmas.txt", "r")



Labelfile = f.read()
Namefile = f1.read()


LableTemp=[]
NameTemp=[]


for i in range(21843):#21843
    print(i)
    xx = Labelfile.split("\n", i+1)
    LableTemp.append(xx[i])
    #print(LableTemp[i])  
    
    yy = Namefile.split("\n", i+1)
    NameTemp.append(yy[i])
    #print(NameTemp[i]) 
    
Save_Mean_text_feature=[]

for i in range(21843):#21843
    print(i)
    temp = NameTemp[i].split(",", 1)
    count = 1
    L = len(temp)
    while L > 1:
        temp1 = temp[1]
        temp = temp1.split(",", 1)
        L = len(temp)
        count = count+1

    text_features=[]
    Mean_text_feature=[]
    temp = NameTemp[i].split(",", 1)
    for j in range(count):
        text = open_clip.tokenize([temp[0]])
        if (count-j) >1:
            temp = temp[1].split(",", 1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features.append(model.encode_text(text))
           # Mean_text_feature = Mean_text_feature + model.encode_text(text)
    
    Mean_text_feature = text_features[0] 
    for j in range(1,count):     
        Mean_text_feature = Mean_text_feature + text_features[j]
        
    Mean_text_feature = Mean_text_feature/count
    Mean_text_feature.Label= LableTemp[i]
    
    sample = {}
    sample['text_features'] = Mean_text_feature
    sample['Label'] = LableTemp[i]
    sample['Name'] = NameTemp[i]
    
    Save_Mean_text_feature.append(sample)

#with open('file.txt','w') as data: 
 #     data.write(str(Save_Mean_text_feature))
import numpy as np
np.save ("Save_Mean_text_feature.npy", Save_Mean_text_feature)
    

temp = Save_Mean_text_feature[0]

data = np.load("Save_Mean_text_feature.npy",allow_pickle=True)

print(temp)

temp = data[0]
print(temp)
#-------------------------------------------------------------------------------
f2 = open("imagenet1000_clsid_to_labels.txt", "r")
imagenet = f2.read()
imagenet1000temp=[]
LableTempimage1000=[]
for i in range(1000):#1000
 #   print(i)
    xx = imagenet.split("\n", i+1)
    imagenet1000temp.append(xx[i])
   # print(imagenet1000temp[i])  

Save_Mean_text_feature_image1000=[]
for i in range(1000):#1000
    print(i)
    tempimage = imagenet1000temp[i].split(":", 1)
    LableTempimage1000.append(tempimage[0])
    
    temp = tempimage[1].split(",", 1)
    count = 0
    L = len(temp)
    while L > 1:
        temp1 = temp[1]
        temp = temp1.split(",", 1)
        L = len(temp)
        count = count+1

    text_features=[]
    Mean_text_feature=[]
    temp = tempimage[1].split(",", 1)
    for j in range(count):
        text = open_clip.tokenize([temp[0]])
        if (count-j) >1:
            temp = temp[1].split(",", 1)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features.append(model.encode_text(text))
           # Mean_text_feature = Mean_text_feature + model.encode_text(text)


    Mean_text_feature = text_features[0] 
    for j in range(1,count):     
        Mean_text_feature = Mean_text_feature + text_features[j]
        
    Mean_text_feature = Mean_text_feature/count

    
    sample = {}
    sample['text_features'] = Mean_text_feature
    sample['Label'] = tempimage[0]
    sample['Name'] = tempimage[1]
    
    Save_Mean_text_feature_image1000.append(sample)
  
temp = Save_Mean_text_feature_image1000[0]
print(temp)
np.save ("Save_Mean_text_feature_image1000.npy", Save_Mean_text_feature_image1000)
    
data = np.load("Save_Mean_text_feature_image1000.npy",allow_pickle=True)

temp = data[0]
print(temp)
