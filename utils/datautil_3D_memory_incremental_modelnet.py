import os, sys, math, random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from path import Path
import torch
import torchvision.transforms as Tr
import pandas as pd
from configs.modelnet_info import task_ids_total as tid
from configs.modelnet_info import len_cls


"""#Preprocess"""
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



"""#custom Dataset"""
class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train"):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        # {'airplane': 0, ....}
        self.classes = {folder: i for i, folder in enumerate(folders)}
        # print(self.classes)
        self.files = []
        self.file_class_count = {category: 0 for category in self.classes.keys()}
        # example of self.files: 
        # {'pcd_path': Path('E:/NSU/sfr1/3DV21/code_ModelNet40/content/ModelnetNew\\xbox\\test\\xbox_0122.pt'), 'category': 'xbox', 'name': 'xbox_0122'}
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pt'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    sample['name'] = file[:-3]
                    self.files.append(sample)
                    self.file_class_count[category]+=1
        # print(self.file_class_count)
        # random.shuffle(self.files)
        del self.file_class_count

    def __len__(self):
        return len(self.files)

class iPointCloudData(PointCloudData):
    def __init__(self, root, task_num, classes, memory_classes, memory, folder, fewshot=0, sem_file=None, transform=None, phase=None):
      super(iPointCloudData, self).__init__(root_dir=root, folder=folder) 
      if not isinstance(classes, list):
        classes= [classes]
      self.phase=phase
      self.task_num=task_num
      self.fewshot = 0 if self.task_num==0 else fewshot
      # print("Memory:", memory, ', Folder:', folder)
      self.folder=folder
      if self.folder=='train':
        # self.class_mapping={c:i for i,c in enumerate(classes)}
        if self.task_num==0:
          self.class_mapping={c:i for i,c in enumerate(classes)}
          # print('train_0_classes: ',self.class_mapping)
        elif self.task_num==1:
          self.class_mapping={c:i+20 for i,c in enumerate(classes)}
          # print('train_1_classes: ',self.class_mapping)
        elif self.task_num==2:
          self.class_mapping={c:i+25 for i,c in enumerate(classes)}
          # print('train_2_classes: ',self.class_mapping)
        elif self.task_num==3:
          self.class_mapping={c:i+30 for i,c in enumerate(classes)} 
          # print('train_3_classes: ',self.class_mapping)
        elif self.task_num==4:
          self.class_mapping={c:i+35 for i,c in enumerate(classes)}   
          # print('train_4_classes: ',self.class_mapping) 

        # print('Train:Num_classes of task {} are:{}'.format(self.task_num,self.class_mapping.keys()))
      elif self.folder=='test':
        self.class_mapping={c:i for i,c in enumerate(classes)}
        # print('Test:Num_classes of task {} are:{}'.format(self.task_num,self.class_mapping.values()))

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
          #######
          if self.folder=='test':         
            if 0<=(self.class_mapping[self.classes[self.files[i]['category']]])<=19:
              task_label.append(0)
            elif 20<=(self.class_mapping[self.classes[self.files[i]['category']]])<=24:
              task_label.append(1)
            elif 25<=(self.class_mapping[self.classes[self.files[i]['category']]])<=29:
              task_label.append(2)
            elif 30<=(self.class_mapping[self.classes[self.files[i]['category']]])<=34:
              task_label.append(3)
            elif 35<=(self.class_mapping[self.classes[self.files[i]['category']]])<=39:
              task_label.append(4) 
          # print(len(task_label))  

      # print(len(pointcloud), len(classes), train_class_file_count) 
      
      if self.phase=='training' and self.folder=="train":
        print("len_data with_Out_mem: ",len(labels))


#########################################ADD Memory
      if memory_classes:
        for j in range(self.task_num):
          for i in range(len(memory[j]['pcdpath'])):
              if memory[j]['label'][i] in memory_classes[j]:
                  pointcloud.append(memory[j]['pcdpath'][i])
                  labels.append(memory[j]['label'][i])
                  names.append(memory[j]['name'][i]) 
                  flag_task.append(1)
  
        print('len_data with memory',len(labels))
        
  #####################################################################
      self.pointcloud= pointcloud   #adress of data of task 
      self.labels = labels
      self.names=names
      self.task_label=task_label
      self.flag_task=flag_task
      # print(self.names, self.labels)
###### add sem
      # tmp_sem_classes = []
      # if memory_classes:
      #   for j in range(self.task_num):
      #     for l in memory[j]['class_label']:
      #       # if l not in memory[j]['class_label']:
      #       tmp_sem_classes.append(l)
      # tmp_sem_classes.extend(classes)
      # sem_classes = []
      # for i in tmp_sem_classes:
      #   if i not in sem_classes:
      #     sem_classes.append(i)
      # wordvector = sio.loadmat(os.path.join(root, sem_file))
      # w2v = wordvector['word']
      # self.sem = w2v[sem_classes,:]
      # print('(',self.folder,') sem shape:', self.sem.shape, sem_classes, len(np.unique(labels)))

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
      pointclouds,labels,names,task = pointclouds,self.labels[index],self.names[index] ,self.flag_task[index]
      class_label = self.classes[names.rsplit('_',1)[0]]

      if self.folder=="test" :
        task_la=self.task_label[index] 
        return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'task_label':task_la, 'class_label': class_label}
      else:
        return {'pointclouds':pointclouds,'labels':labels,'pcd_path':pcd_path,'names':names,'flag_memory':task, 'class_label': class_label}



"""#DataLoader"""
class DatasetGen(object):
    """docstring for DatasetGen"""
    def __init__(self, args, root, fewshot=0):
        super(DatasetGen, self).__init__()
        self.root = root
        self.fewshot = fewshot
        self.batch_size = args.batch_size
        self.sem_file = args.sem_file
        # self.pc_valid=args.pc_valid
        self.num_workers = args.workers
        self.pin_memory = False #True 
        self.num_tasks = int(args.ntasks)
        self.num_classes =args.nclasses
        self.use_memory = args.use_memory
        self.num_samples = args.num_samples 
        self.inputsize = [1024,3]
        self.do_herding = args.herding
        self.transformation = Tr.Compose([           
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
        self.default_transforms=Tr.Compose([
                                Normalize(),
                                ToTensor()
                              ])  
      
        self.task_memory={} 
        self.herding_memory={}
        self.counter={}
        ###########   ADD MEMORY
        for i in range(self.num_tasks):
            self.task_memory[i] = {}
            self.herding_memory[i] = {}
            self.counter[i]={}
            self.task_memory[i]['name'] = []
            self.task_memory[i]['pcdpath'] = []
            self.task_memory[i]['label'] = []
            self.task_memory[i]['class_label'] = []
            self.herding_memory[i]['name'] = []
            self.herding_memory[i]['pcdpath'] = []
            self.herding_memory[i]['label'] = []
            self.herding_memory[i]['class_label'] = []
            self.counter[i]['label']=[]
            self.counter[i]['name'] = []
            self.counter[i]['pcdpath'] = []
            self.counter[i]['class_label'] = []

        # self.task_split= np.split(np.random.permutation(self.num_classes),2)
        # self.d=[list(arr) for arr in (self.task_split)]
        # self.task_splits=np.split(np.random.permutation(self.task_split[1]),4)
        # self.e= [list(arr) for arr in self.task_splits]
        self.task_ids_total=tid
        # print(self.task_ids_total)
    
    def get(self, task_id, phase):
        self.dataloaders = {}
        self.dataloaders[task_id] = {}
    
        if task_id == 0:
            memory_classes = None
            memory=None
        else:
            memory_classes=[[i for i in range(20)],[i for i in range(20,25)],[i for i in range(25,30)],[i for i in range(30,35)],[i for i in range(35,40)]]
            memory = self.task_memory

        task_id_test=task_id
        task_ids_test=[]
        task_ids=[list(arr) for arr in self.task_ids_total]
        for i in range(task_id_test + 1):       
            task_ids_test=self.task_ids_total[task_id_test]+task_ids_test
            task_id_test = task_id_test - 1

        self.train_set = {}
        self.test_set = {}     
        sys.stdout.flush()

        # print(task_ids)

        # task_ids = [[i for i in range(0,40)], [i for i in range(0,40)], [i for i in range(0,40)], [i for i in range(0,40)], [i for i in range(0,40)]]
        # # print(task_ids)
        # task_ids_test = [i for i in range(0,40)]

        self.train_set[task_id] = iPointCloudData(root=self.root, classes=task_ids[task_id], memory_classes=memory_classes, 
                                                sem_file=self.sem_file, memory=memory, task_num=task_id, folder="train", 
                                                transform=self.default_transforms, phase=phase, fewshot=self.fewshot)
        self.test_set[task_id] = iPointCloudData(root=self.root, classes=task_ids_test, memory_classes=None, memory=None, 
                                                sem_file=self.sem_file, task_num=task_id, folder='test', 
                                                transform=self.default_transforms, phase=phase, fewshot=0)
        
        train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=1, num_workers=self.num_workers,
                                                    pin_memory=self.pin_memory, shuffle=True)
        self.dataloaders[task_id]['train'] = train_loader  
        self.dataloaders[task_id]['test'] = test_loader 
        
        # self.dataloaders[task_id]['sem_train'] = torch.from_numpy(self.train_set[task_id].sem).float()
        # self.dataloaders[task_id]['sem_test'] = torch.from_numpy(self.test_set[task_id].sem).float()
        
        if phase=='training':     
            print ('Task ID: {} -> {}'.format(task_id,task_ids[task_id]))
            # print ("Task Clases:", task_ids[task_id], ", Memory classes:", memory_classes)
            print ("Training set size:   {} pointcloud of {}x{}".format(len(train_loader.dataset),self.inputsize[0],self.inputsize[1]))
            print ("Test set size:       {} pointcloud of {}x{}".format(len(test_loader.dataset),self.inputsize[0],self.inputsize[1])) 

        # memory has been updated for each task initialization
        if self.use_memory and self.num_samples > 0 :
            self.update_memory(task_id,phase)
        return self.dataloaders
    
    def update_memory(self, task_id, phase): 
        # num_samples_per_class=1 
        data_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=1)  
        randind = torch.randperm(len(data_loader.dataset))  
        for ind in randind:
            self.counter[task_id]['label'].append(data_loader.dataset[ind]['labels'])
            self.counter[task_id]['name'].append(data_loader.dataset[ind]['names'])
            self.counter[task_id]['pcdpath'].append(data_loader.dataset[ind]['pcd_path'])
            self.counter[task_id]['class_label'].append(data_loader.dataset[ind]['class_label'])
        
        a=pd.DataFrame(self.counter[task_id])
        if self.do_herding: # number of samples that you want    
          a=a.groupby(by='label' ,as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, len(array), False),:])
        else:
          a=a.groupby(by='label' ,as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, self.num_samples, False),:])
          
        a=a.to_numpy()
        self.task_memory[task_id]['label']=a[:,0]
        self.task_memory[task_id]['name']=a[:,1]
        self.task_memory[task_id]['pcdpath']=a[:,2]
        self.task_memory[task_id]['class_label']=a[:,3]
        # print(self.task_memory[task_id]['label'])
        if phase=='training':               
            print ('Task Memory is updated by adding {} images'.format(len(self.task_memory[task_id]['label'])))


class argument():
    dataset_path =  "dataset/modelnet_scanobjectnn/"
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
