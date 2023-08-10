import open_clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# read text file
f = open('storage.googleapis.com_bit_models_imagenet21k_wordnet_ids.txt', 'r')
f1 = open('storage.googleapis.com_bit_models_imagenet21k_wordnet_lemmas.txt', 'r')

Labelfile = f.read()
Namefile = f1.read()


LableTemp=[]
NameTemp=[]


for i in range(21843):#21843
    print(i)
    xx = Labelfile.split("\n", i+1)
    LableTemp.append(xx[i])
    
    yy = Namefile.split("\n", i+1)
    NameTemp.append(yy[i])
    
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
    Save_Mean_text_feature.append(Mean_text_feature)



temp = Save_Mean_text_feature[2]
  
print(len(Save_Mean_text_feature))
