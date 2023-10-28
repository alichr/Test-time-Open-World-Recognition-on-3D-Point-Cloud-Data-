import torch, sys, h5py, numpy as np, pandas as pd, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys
import pandas as pd

def load_scanobjectnn_data(partition):
    BASE_DIR='E:/NSU/sfr1/3D/dataset/scanobjectnn/h5_files/main_split_nobg/'
   
    all_data = []
    all_label = []

    h5_name = BASE_DIR + partition + '_objectdataset.h5'
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


"""#custom dataset"""
class iScanObjectNN(Dataset):
    def __init__(self,num_points,partition,classes,task_num,memory_classes,memory,phase=None):    
      self.data, self.label = load_scanobjectnn_data(partition)
      self.num_points = num_points
      self.partition = partition
      self.task_num=task_num
      pointcloud=[]
      labels=[]
      task_label=[]
      flag_task=[]
      if not isinstance(classes, list):
          classes= [classes]
      if self.partition=='training':
        self.class_mapping={c:i+self.task_num*3 for i,c in enumerate(classes)}
      elif self.partition=='test':
        self.class_mapping={c:i for i,c in enumerate(classes)}  

      for i in range (len(self.label)):
        if self.label[i] in classes:
          pointcloud.append(self.data[i][:self.num_points])   
          labels.append(self.class_mapping[self.label[i]])
          flag_task.append(0)
          if self.partition=='test': 
            if 0<=(self.class_mapping[self.label[i]])<=2:
              task_label.append(0)
            elif 3<=(self.class_mapping[self.label[i]])<=5:
              task_label.append(1)
            elif 6<=(self.class_mapping[self.label[i]])<=8:
              task_label.append(2)
            elif 9<=(self.class_mapping[self.label[i]])<=11:
              task_label.append(3)
            elif 12<=(self.class_mapping[self.label[i]])<=14:
              task_label.append(4) 

      if phase=='training' and self.partition=='training':
        print("len_data with_Out_mem: ",len(labels))
     
      if memory_classes:
        for j in range(self.task_num):
          for i in range(len(memory[j]['pointclouds'])):
              if memory[j]['label'][i] in memory_classes[j]:
                  pointcloud.append(memory[j]['pointclouds'][i])
                  labels.append(memory[j]['label'][i])
                  flag_task.append(1)

        print('len_data with memory',len(labels))                
      self.pointcloud=pointcloud
      self.labels=labels
      self.flag_task=flag_task
      self.task_label=task_label
    

    def __getitem__(self, item):
        pointclouds = self.pointcloud[item]
        labels = self.labels[item]
        task=self.flag_task[item]
        if self.partition == 'training':
            pointclouds = translate_pointcloud(pointclouds)
        if self.partition=='test':
          task_la=self.task_label[item]
          return {'pointclouds':pointclouds,'labels':labels,'task_label':task_la}
        else:  
          return {'pointclouds':pointclouds,'labels':labels,'flag_memory':task}

    def __len__(self):
        return len(self.pointcloud)



"""#DatasetGen"""
class DatasetGen(object):
    """docstring for DatasetGen"""
    def __init__(self, args):
        super(DatasetGen, self).__init__()
        self.batch_size=args.batch_size
        self.num_workers = args.workers
        self.pin_memory = True 
        self.num_tasks = args.ntasks
        self.num_classes =args.nclasses
        self.use_memory = args.use_memory
        self.num_samples=args.num_samples
        self.inputsize = [1024,3]
        self.task_memory={} 
        self.counter={}
###########   ADD MEMORY
        for i in range(self.num_tasks):
          self.task_memory[i] = {}
          self.counter[i]={}
          self.task_memory[i]['pointclouds'] = []
          self.task_memory[i]['label'] = []
          self.counter[i]['label']=[]     
          self.counter[i]['pointclouds'] = []

        self.task_split= np.split(np.random.permutation(self.num_classes),args.ntasks)
        self.task_ids_total=[list(arr) for arr in (self.task_split)]
    
    def get(self, task_id,phase):
      self.dataloaders = {}
      self.dataloaders[task_id] = {}
    

###############   ADD MEMORY
      if task_id ==0 or self.use_memory=='No':
        memory_classes = None
        memory=None
      else:
        memory_classes=[list(arr) for arr in np.split(np.arange(self.num_classes),args.ntasks)]
        memory = self.task_memory

################
      task_id_test=task_id
      task_ids_test=[]
      task_ids=[list(arr) for arr in self.task_ids_total]
      for i in range(task_id_test + 1):       
          task_ids_test=self.task_ids_total[task_id_test]+task_ids_test
          task_id_test = task_id_test - 1

      self.train_set = {}
      self.test_set = {}     
      sys.stdout.flush()

      self.train_set[task_id] =iScanObjectNN (num_points=1024,partition='training',classes=task_ids[task_id],task_num=task_id,memory_classes=memory_classes,memory=memory,phase=phase)
      self.test_set[task_id] =iScanObjectNN(num_points=1024 ,partition='test',classes=task_ids_test,task_num=task_id,memory_classes=None,memory=None,phase=phase)
    
      train_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=32, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory,shuffle=True)
      test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=16, num_workers=self.num_workers,
                                                pin_memory=self.pin_memory, shuffle=True)
          
      self.dataloaders[task_id]['train'] = train_loader  
      self.dataloaders[task_id]['test'] = test_loader 
      if phase=='training':     
        print ( 'Task ID: -{}-{}'.format(task_id,task_ids[task_id]))
        print ("Training set size:   {} pointcloud of {}x{}".format(len(train_loader.dataset),self.inputsize[0],self.inputsize[1]))
        print ("Test set size:       {} pointcloud of {}x{}".format(len(test_loader.dataset),self.inputsize[0],self.inputsize[1])) 
      if self.use_memory == 'yes' and self.num_samples > 0 :
            self.update_memory(task_id,phase)
      return self.dataloaders
    def update_memory(self, task_id,phase): 
      data_loader = torch.utils.data.DataLoader(self.train_set[task_id], batch_size=1)  
      randind = torch.randperm(len(data_loader.dataset))  
      for ind in randind:
          self.counter[task_id]['label'].append(data_loader.dataset[ind]['labels'])
          self.counter[task_id]['pointclouds'].append(data_loader.dataset[ind]['pointclouds'])
      df=pd.DataFrame(self.counter[task_id])
      Samplesize = self.num_samples #number of samples that you want       
      a=df.groupby(by='label' ,as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])
      a=a.to_numpy()
      self.task_memory[task_id]['label']=a[:,0]
      self.task_memory[task_id]['pointclouds']=a[:,1]
      if phase=='training':               
        print ('Memory updated by adding {} images'.format(len(self.task_memory[task_id]['label'])))
