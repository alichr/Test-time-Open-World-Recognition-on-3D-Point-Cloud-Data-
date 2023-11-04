import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import csv 
import glob
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from path import Path
import math
import random
from torchvision import transforms, utils,datasets, models
import sys

# print current directory



########### constant
nb_cl_fg = 44
len_cls = [44, 49, 54, 59] 
model_heads = [44, 5, 5, 5]
task_ids_total=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
                 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], [44, 45, 46, 47, 48], [49, 50, 51, 52, 53], [54, 55, 56, 57, 58]]
tid = task_ids_total

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud

class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)

class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train"):
        self.root_dir = root_dir
        tmp = os.listdir(root_dir)
        tmp.remove(".DS_Store")
        folders = sorted([int(dir) for dir in tmp])
        folders = [str(i) for i in folders]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.files = []
        self.file_class_count = {category: 0 for category in self.classes.keys()}
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pt'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    sample['name'] = category
                    self.files.append(sample)
                    self.file_class_count[category]+=1
        del self.file_class_count

    def __len__(self):
        return len(self.files)

class iPointCloudData(PointCloudData):
    def __init__(self, root, task_num, classes, folder, fewshot=0,  transform=None, phase=None):
      super(iPointCloudData, self).__init__(root_dir=root, folder=folder)
      if not isinstance(classes, list):
        classes= [classes]
      self.phase=phase
      self.task_num=task_num
      self.fewshot = 0 if self.task_num==0 else fewshot
      self.folder=folder
      if self.folder=='train':
        if self.task_num==0:
          self.class_mapping={c:i for i,c in enumerate(classes)}
        else:
          self.class_mapping={c:i+len_cls[self.task_num-1] for i,c in enumerate(classes)}
      elif self.folder=='test':
        self.class_mapping={c:i for i,c in enumerate(classes)}
      self.transforms = transform
      pointcloud=[]
      labels=[]
      names=[]
      flag_task=[]
      task_label=[]

      if self.fewshot>0 and self.folder=='train' and self.phase=='training':
        train_class_file_count = {i:0 for i in classes}

      for i in range(len(self.files)):
        if self.classes[self.files[i]['category']] in classes:
          if self.fewshot>0 and self.folder=='train' and self.phase=='training':
            if train_class_file_count[self.classes[self.files[i]['category']]]>=self.fewshot:
              continue
            else:
              train_class_file_count[self.classes[self.files[i]['category']]]+=1

          pointcloud.append(self.files[i]['pcd_path'])
          labels.append(self.class_mapping[self.classes[self.files[i]['category']]])
          names.append(self.files[i]['name'])
          flag_task.append(0)
          if self.folder=='test':
            for k in range(len(len_cls)):
              l=len_cls[k]
              j=0 if k==0 else len_cls[k-1]
              if self.class_mapping[self.classes[self.files[i]['category']]] in [m for m in range(k, l)]:
                task_label.append(k)
      if self.phase=='training' and self.folder=="train":
        print("len_data with_Out_mem: ",len(labels))

      self.pointcloud= pointcloud   #adress of data of task
      self.labels = labels
      self.names=names
      self.task_label=task_label
      self.flag_task=flag_task
    def __len__(self):
        return len(self.pointcloud)

    def __preproc__(self, file):
        pcld = torch.load(file)
        if self.transforms:
          pointclouds = self.transforms(pcld)
        return pointclouds

    def __getitem__(self, index):
        pcd_path = self.pointcloud[index]
        pointclouds = self.__preproc__(pcd_path)
        pointclouds,labels,names,task = pointclouds,self.labels[index],self.names[index],self.flag_task[index]
        class_label = self.names[index]

        if self.folder=="test" :
            task_la=self.task_label[index]
            return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'task_label':task_la,'class_label': class_label}
        else:
            return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'flag_memory':task,'class_label': class_label}


############################# Class Dataset Generator
class DatasetGen(object):
    def __init__(self, args, root, fewshot=0):
        super(DatasetGen, self).__init__()
        self.root = root
        self.fewshot = fewshot
        self.batch_size = args.batch_size
        self.sem_file = None
        self.num_workers = args.workers
        self.pin_memory = False #True
        self.num_tasks = args.ntasks
        self.num_classes =args.nclasses
        self.num_samples = args.num_samples
        self.inputsize = [1024,3]
        self.transformation = transforms.Compose([
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
        self.default_transforms=transforms.Compose([
                                Normalize(),
                                ToTensor()
                              ])


        self.task_ids_total=tid

    def get(self, task_id, phase):
        self.dataloaders = {}
        self.dataloaders[task_id] = {}



        task_id_test=task_id
        task_ids_test=[]
        task_ids=[list(arr) for arr in self.task_ids_total]
        for i in range(task_id_test + 1):
            task_ids_test=self.task_ids_total[task_id_test]+task_ids_test
            task_id_test = task_id_test - 1

        self.train_set = {}
        self.test_set = {}
        sys.stdout.flush()

        self.train_set[task_id] = iPointCloudData(root=self.root, classes=task_ids[task_id], 
                                                 task_num=task_id, folder="train",
                                                transform=self.default_transforms, phase=phase, fewshot=self.fewshot)

        self.test_set[task_id] = iPointCloudData(root=self.root, classes=task_ids_test, 
                                                task_num=task_id, folder='test',
                                                transform=self.default_transforms, phase=phase, fewshot=0)

        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory, shuffle=True)
        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['test'] = test_loader

        if phase=='training':
            print ('Task ID: {} -> {}'.format(task_id,task_ids[task_id]))
            print ("Training set size:   {} pointcloud of {}x{}".format(len(train_loader.dataset),self.inputsize[0],self.inputsize[1]))
            print ("Test set size:       {} pointcloud of {}x{}".format(len(test_loader.dataset),self.inputsize[0],self.inputsize[1]))
        return self.dataloaders
    

class custom_data_set(Dataset):
    def __init__(self, pcd_path ,transform,labels ):

      self.transforms = transform
      pointcloud=pcd_path
      labels=labels
      self.pointcloud= pointcloud   #adress of data of task
      self.labels = labels
    def __len__(self):
        return len(self.pointcloud)

    def __preproc__(self, file):
        pcld = torch.load(file)
        if self.transforms:
          pointclouds = self.transforms(pcld)
        return pointclouds

    def __getitem__(self, index):
        pcd_path = self.pointcloud[index]
        # print(pcd_path)
        pointclouds = self.__preproc__(pcd_path)
        # print(pointclouds.size())
        pointclouds,labels = pointclouds,self.labels[index]
        return {'pointclouds':pointclouds,'labels':labels}    


transformation = transforms.Compose([
            Normalize(),
            RandRotation_z(),
            RandomNoise(),
            ToTensor()
            ])


class argument():
    dataset_path =  "dataset/FSCIL/shapenet_scanobjectnn"
    #r'C:\Users\24723266\OneDrive - UTS\Documents\GitHub\test\modelnet_scanobjectnn'
    # 
    lr = 1e-3
    wd = 1e-6
    epochs= 1
    nclasses = len_cls[-1]
    seed = 42
    ntasks = len(len_cls)
    batch_size = 16
    workers = 4
    # use_memory = True
    use_memory = False
    num_samples = 0 # number of samples - 20
    feature_dim = 1024
    fewshot = 1

  
  

if __name__ == "__main__":
#args=argument()
  path=Path(argument.dataset_path)
  print(path)
  dataloader=DatasetGen(argument, root=path, fewshot=argument.fewshot)
  t = 0# 0:1-25 , 1:26-37
  
  
  dataset = dataloader.get(t,'training')
  trainloader = dataset[t]['train']
  print(trainloader.__len__())
  testloader = dataset[t]['test'] 
  
  Train_Data = next(iter(trainloader))
  print(Train_Data['labels'])
  print(Train_Data['pcd_path'][10])
  print(torch.load(Train_Data['pcd_path'][10]))
  LengthDataTrain = trainloader.batch_sampler.sampler.num_samples
  
  Test_Data = next(iter(testloader))
  print(Test_Data['labels'])
  print(Test_Data['pcd_path'][10])
  print(torch.load(Test_Data['pcd_path'][10]))
  LengthDataTest = testloader.batch_sampler.sampler.num_samples
  
  
  
  LengthDataTrain2 = len(trainloader.dataset.files)
  #print(len(trainloader.dataset.files))
  LengthDataTest = testloader.batch_sampler.sampler.num_samples
# Creat List for Shuffling -------------------------------------
  my_listCount = []
  
  for ii in range(LengthDataTrain):
    my_listCount.append(ii)
   
  random.shuffle(my_listCount)

# Save Shuffle List!
  with open('randomlist.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(my_listCount)
 
 
 
 ##---------------------------------------------------------------
 #TranData 
  objects_list2 = []
  for ii in range(2):#int(LengthData/argument.batch_size)):
    Numbmat =[]
    objects_list = []
    for jj in range(argument.batch_size):
      Numbmat.append(my_listCount[jj+ii*argument.batch_size])
      
      pcld2 = torch.load(trainloader.dataset.files[Numbmat[jj]]['pcd_path'])
      label1=trainloader.dataset.labels[Numbmat[jj]]
      name11 = trainloader.dataset.names[Numbmat[jj]]
      sample = {}
      sample['Label'] = label1
      sample['Class'] = name11
      sample['Data'] = pcld2
      sample['NumberSort']=Numbmat[jj]
      obj = sample  # Replace this with the actual object creation logic
      objects_list.append(obj)
    objects_list2.append(objects_list)
#-----------------------------------------------------------------------
  # Creat List for Shuffling -------------------------------------
  my_listCountTest = []
  
  for ii in range(LengthDataTest):
    my_listCountTest.append(ii)
   
  random.shuffle(my_listCountTest)

# Save Shuffle List!
  with open('randomlisttest.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(my_listCountTest)
    
 #TestData 
  objects_listTest2 = []
  for ii in range(2):#int(LengthDataTest/argument.batch_size)):
    Numbmat =[]
    objects_listTest = []
    for jj in range(argument.batch_size):
      Numbmat.append(my_listCountTest[jj+ii*argument.batch_size])
      
      pcld2 = torch.load(testloader.dataset.files[Numbmat[jj]]['pcd_path'])
      label1=testloader.dataset.labels[Numbmat[jj]]
      name11 = testloader.dataset.names[Numbmat[jj]]
      sample = {}
      sample['Label'] = label1
      sample['Class'] = name11
      sample['Data'] = pcld2
      sample['NumberSort']=Numbmat[jj]
      obj = sample  # Replace this with the actual object creation logic
      objects_listTest.append(obj)
    objects_listTest2.append(objects_listTest)
 
 ##---------------------------------------------------------------
  print(objects_list[10]['Class'])
      

  
  
  
  


    
