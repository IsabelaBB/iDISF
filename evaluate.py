
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage.morphology import distance_transform_edt
#from core import init_model,predict
import random
from idisf import iDISF_scribbles
from matplotlib import pyplot as plt
import pandas as pd
import openpyxl
import xlsxwriter
import time

datasets_path='../../mestrado/disciplinas/AM/Seminario_I/Res2Net-fcanet-master/'
idisf_path='../iDISF/'

########################################[ Dataset ]########################################
#for general dataset format
class Dataset():
    def __init__(self,dataset_path,img_folder='img',gt_folder='gt',threshold=128,ignore_label=None):
        self.index,self.threshold,self.ignore_label = 0,threshold,ignore_label
        dataset_path=Path(dataset_path)
        self.img_files = sorted((dataset_path/img_folder).glob('*.*'))
        self.gt_files = [ next((dataset_path/gt_folder).glob(t.stem+'.*')) for t in self.img_files]
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.img_files)
    def __next__(self):
        if self.index > len(self) - 1:raise StopIteration
        img_src = np.array(Image.open(self.img_files[self.index]))
        gt_src = np.array(Image.open(self.gt_files[self.index]))
        gt = gt_src[:,:,0] if gt_src.ndim==3 else gt_src
        gt = np.uint8(gt>=self.threshold)
        if self.ignore_label is not None: gt[gt_src==self.ignore_label]=255
        self.index += 1
        return img_src,gt

#special for PASCAL_VOC2012
class VOC2012():
    def __init__(self,dataset_path):
        self.index = 0
        dataset_path=Path(dataset_path)
        #with open(dataset_path/'ImageSets'/'Segmentation'/'val.txt') as f:
        with open(dataset_path/'ImageSets'/'Segmentation'/'trainval.txt') as f:
            val_ids=sorted(f.read().splitlines())

        self.img_files,self.gt_files,self.instance_indices=[],[],[]
        print('Preprocessing!')
        for val_id in tqdm(val_ids):
            gt_ins_set=  sorted(set(np.array(Image.open( dataset_path/'SegmentationObject'/(val_id+'.png'))).flat))
            for instance_index in gt_ins_set:
                if instance_index not in [0,255]:
                    self.img_files.append(  dataset_path/'JPEGImages'/(val_id+'.jpg'))
                    self.gt_files.append(  dataset_path/'SegmentationObject'/(val_id+'.png'))
                    self.instance_indices.append(instance_index)
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.img_files)
    def __next__(self):
        if self.index > len(self) - 1:raise StopIteration
        img_src = np.array(Image.open(self.img_files[self.index]))
        gt_src = np.array(Image.open(self.gt_files[self.index]))
        gt=np.uint8(gt_src==self.instance_indices[self.index])
        gt[gt_src==255]=255
        self.index += 1
        return img_src,gt


########################################[ Evaluation ]########################################
#robot user strategy
def get_next_anno_point(pred, gt, seq_points):
    fndist_map=distance_transform_edt(np.pad((gt==1)&(pred==0),((1,1),(1,1)),'constant'))[1:-1, 1:-1]
    fpdist_map=distance_transform_edt(np.pad((gt==0)&(pred==255),((1,1),(1,1)),'constant'))[1:-1, 1:-1]
    fndist_map[seq_points[:,1],seq_points[:,0]],fpdist_map[seq_points[:,1],seq_points[:,0]]=0,0
    
    [usr_map,if_pos] = [fndist_map, 1] if fndist_map.max() > fpdist_map.max() else [fpdist_map, 0]
    [y_mlist, x_mlist] = np.where(usr_map == usr_map.max())
    
    pt_next=(x_mlist[0],y_mlist[0],if_pos)
    return pt_next

datasets_kwargs={
    'GrabCut' :{'dataset_path':datasets_path+'dataset/GrabCut' ,'img_folder':'data_GT','gt_folder':'boundary_GT','threshold':128,'ignore_label':128 },
    'Berkeley':{'dataset_path':datasets_path+'dataset/Berkeley','img_folder':'images' ,'gt_folder':'masks'      ,'threshold':128,'ignore_label':None},
    'DAVIS'   :{'dataset_path':datasets_path+'dataset/DAVIS'   ,'img_folder':'img'    ,'gt_folder':'gt'         ,'threshold':0.5,'ignore_label':None},
    'VOC2012' :{'dataset_path':datasets_path+'dataset/VOC2012'},
}
default_miou_targets={'GrabCut':0.90,'Berkeley':0.90,'DAVIS':0.90,'VOC2012':0.85}

def getMarkers(markers, nextPoint, markers_sizes):
    (x,y,if_pos) = nextPoint
    
    num_neg_points = 0

    if(len(markers) == 0):
        num_pos_points = 0;
    else:
        num_pos_points = markers_sizes[0]

        if(len(markers_sizes) == 2):
            num_neg_points = markers_sizes[1]

    if(if_pos):
        markers_pos = markers[:num_pos_points]
        markers = markers_pos + [[x,y]] + markers[num_pos_points:]
        num_pos_points += 1
    else:
        markers = markers + [[x,y]]
        num_neg_points += 1
    
    markers_sizes = []
    markers_sizes.append(num_pos_points)

    if(num_neg_points > 0):
        markers_sizes.append(num_neg_points)
    
    return markers, markers_sizes


def eval_dataset(model, dataset, max_point_num=20, record_point_num=20,if_sis=False,miou_target=None,if_cuda=True):
    global datasets_kwargs, default_miou_targets
    if dataset in datasets_kwargs:
        dataset_iter= VOC2012(**datasets_kwargs[dataset]) if dataset=='VOC2012' else  Dataset(**datasets_kwargs[dataset]) 
        miou_target = default_miou_targets[dataset] if miou_target is None else miou_target
    else:
        dataset_iter=Dataset(dataset_path='dataset/{}'.format(dataset)) 
        miou_target = 0.85 if miou_target is None else miou_target

    NoC,mIoU_NoC=0,[0]*(record_point_num+1)

    count = 0
    
    for img,gt in tqdm(dataset_iter):
        pred = np.zeros_like(gt)
        seq_points=np.empty([0,3],dtype=np.int64)
        if_get_target=False
        
        markers = []
        markers_sizes = []
        objMarkers = 1
        miou_img = []

        n0 = 200
        iterations = 3
        function = 1
        c1 = 0.7
        c2 = 1.0
        segm_method = 1 #{1:runiDISF_scribbles_rem, 2:runiDISF}
        bordersValue = 0
        
        for point_num in range(1, max_point_num+1):
            pt_next = get_next_anno_point(pred, gt, seq_points)
            seq_points=np.append(seq_points,[pt_next],axis=0)
            getValidPoint(gt, 1, 1, 1, 1, 1, True, seq_points, 1, 1)

            markers, markers_sizes = getMarkers(markers, pt_next, markers_sizes)
            
            markers_np = np.array(markers)
            marker_sizes_np = np.array(markers_sizes)
            
            pred, border_img = iDISF_scribbles(img, n0, iterations, markers_np, marker_sizes_np, objMarkers, function, c1, c2, segm_method, bordersValue)
            pred = 255 - ((pred-1) * (255/(np.max(pred)-1)) )

            """
            pred = predict(model,img,seq_points,if_sis=if_sis,if_cuda=if_cuda)
            """
            
            miou = (((pred==255)&(gt==1)).sum())/(((pred==255)|(gt==1))&(gt!=255)).sum()         
            
            if point_num <= record_point_num:
                mIoU_NoC[point_num]+=miou
            if (not if_get_target) and (miou >= miou_target or point_num==max_point_num):
                NoC+=point_num
                if_get_target=True
            if if_get_target and  point_num >= record_point_num: break
            
            miou_img.append(miou)

        print('iou:',miou_img)
        #count +=1 
        #if count == 10: break
        
    print('dataset: [{}] {}:'.format(dataset,'(SIS)'if if_sis else ' '))
    print('--> mNoC : {}'.format(NoC/len(dataset_iter)))
    print('--> mIoU-NoC : {}\n\n'.format(np.array([round(i/len(dataset_iter),3) for i in mIoU_NoC ])))


def getValidPoints(gt, num_pos_points, num_neg_points, pt_next, p1, p2, n1, n2, n3):
    
    fndist_map=distance_transform_edt(gt) # distancia para pixels negativos
    fpdist_map=distance_transform_edt(1 - gt) # distancia para pixels positivos

    seq_points=np.empty([0,3],dtype=np.int64)
    tmp = np.ones(gt.shape)
    markers = []
    markers_sizes = []

    seq_points=np.append(seq_points,[pt_next],axis=0)
    markers, markers_sizes = getMarkers(markers, pt_next, markers_sizes)
    tmp[seq_points[:,1],seq_points[:,0]] = 0
    usr_map=distance_transform_edt(tmp)
    
    for i in range(num_pos_points):
        [y_mlist, x_mlist] = np.where((fndist_map > p1) & (usr_map > p2))

        index = random.randint(0,len(y_mlist)-1)
        pt_next=(x_mlist[index],y_mlist[index],1)
        seq_points=np.append(seq_points,[pt_next],axis=0)
        markers, markers_sizes = getMarkers(markers, pt_next, markers_sizes)

        tmp[seq_points[:,1],seq_points[:,0]] = 0
        usr_map=distance_transform_edt(tmp)


    [y_mlist, x_mlist] = np.where((fpdist_map >= n1) & (fpdist_map <= n2) & (usr_map > n3))

    if(len(y_mlist) > 0 and num_neg_points > 0):
        neg_points = random.sample(range(len(y_mlist)), min(num_neg_points, len(y_mlist)-1))

        for i in neg_points:
            pt_next=(x_mlist[i],y_mlist[i],0)
            seq_points=np.append(seq_points,[pt_next],axis=0)
            markers, markers_sizes = getMarkers(markers, pt_next, markers_sizes)

    """
    plt.imshow(fndist_map, interpolation='nearest')
    plt.show()

    plt.imshow(fpdist_map, interpolation='nearest')
    plt.show()
    
    plt.imshow(usr_map, interpolation='nearest')
    plt.show()
    """
    return markers,markers_sizes


def findBestParams(dataset, maxSearches, miou_target, \
        p1_list=[5, 10, 15, 20], p2_list=[7, 10, 20], \
        n1_list=[15, 40, 60], n2_list=[80], n3_list=[10, 15, 25], \
        n0_list=[100, 200, 500, 750, 1000], iterations_list=[*range(1,11)], \
        function=1, c1_list=[(x/10) for x in range(1,11)], c2_list=[(x/10) for x in range(1,11)]):
    
    segm_method = 1 #{1:runiDISF_scribbles_rem, 2:runiDISF}
    bordersValue = 0
    objMarkers = 1

    global datasets_kwargs, default_miou_targets
    if dataset in datasets_kwargs:
        dataset_iter= VOC2012(**datasets_kwargs[dataset]) if dataset=='VOC2012' else  Dataset(**datasets_kwargs[dataset]) 
        miou_target = default_miou_targets[dataset] if miou_target is None else miou_target
    else:
        dataset_iter=Dataset(dataset_path='dataset/{}'.format(dataset)) 
        miou_target = 0.85 if miou_target is None else miou_target

    
    best_miou = 0.0
    best_params = {'n0':0 ,'iterations':1, 'c1':0.1, 'c2':0.1, 'P1':0, 'P2':0, 'N1':0, 'N2':0, 'N3':0}
    
    info_test = {'test_id':[], 'n0':[] ,'iterations':[], 'c1':[], 'c2':[], 'miou':[], 'P1':[], 'P2':[], 'N1':[], 'N2':[], 'N3':[], 'mtime':[]}
    #info_searches_detailed = []

    #print('dataset: [{}]:'.format(dataset))

    for i in range(maxSearches):
        n0 = random.choice(n0_list)
        iterations = random.choice(iterations_list)
        c1, c2 = random.choice(c1_list), random.choice(c2_list)
        p1, p2 = random.choice(p1_list), random.choice(p2_list)
        n1, n2, n3 = random.choice(n1_list), random.choice(n2_list), random.choice(n3_list)

        info_test['test_id'].append(i)
        info_test['n0'].append(n0), info_test['iterations'].append(iterations)
        info_test['c1'].append(c1), info_test['c2'].append(c2)
        info_test['P1'].append(p1), info_test['P2'].append(p2)
        info_test['N1'].append(n1), info_test['N2'].append(n2), info_test['N3'].append(n3)

        info_detailed_local = {'test_id':[], 'image':[] ,'iou':[], 'time':[], 'pos_points':[], 'neg_points':[]}
        miou_test_acum = 0.0
        time_test_acum = 0.0

        img_id = 0
        for img,gt in tqdm(dataset_iter):
            pred = np.zeros_like(gt)
            
            markers = []
            markers_sizes = []

            num_pos, num_neg = random.randint(1,11), random.randint(0,11)

            pt_next = get_next_anno_point(pred, gt, np.empty([0,3],dtype=np.int64))
            markers, markers_sizes = getValidPoints(gt, num_pos-1, num_neg, pt_next, p1, p2, n1, n2, n3)
        
            info_detailed_local['test_id'].append(i)
            info_detailed_local['image'].append(str(dataset_iter.img_files[img_id]).split('/')[-1])
            
            info_detailed_local['pos_points'].append(markers[:markers_sizes[0]])
            if(len(markers_sizes) > 0):
                info_detailed_local['neg_points'].append(markers[markers_sizes[0]:])
            else:
                info_detailed_local['neg_points'].append([])
            

            markers_np, marker_sizes_np = np.array(markers), np.array(markers_sizes)
            
            start = time.time()
            pred, border_img = iDISF_scribbles(img, n0, iterations, markers_np, marker_sizes_np, objMarkers, function, c1, c2, segm_method, bordersValue)
            end = time.time()
            
            pred = 255 - ((pred-1) * (255/(np.max(pred)-1)) )
            miou = (((pred==255)&(gt==1)).sum())/(((pred==255)|(gt==1))&(gt!=255)).sum()         
            
            info_detailed_local['time'].append(end-start)
            info_detailed_local['iou'].append(miou)
            miou_test_acum+=miou
            time_test_acum+=(end-start)

            img_id +=1
            

        df_unit = pd.DataFrame(info_detailed_local)
        df_unit.to_csv('unit_tests/test_'+str(i)+'.csv', index=False)

        miou_test_acum = round(miou_test_acum/len(dataset_iter),5)
        time_test_acum = round(time_test_acum/len(dataset_iter),5)
        
        info_test['miou'].append(miou_test_acum)
        info_test['mtime'].append(time_test_acum)

        if miou_test_acum > best_miou:
            best_miou = miou_test_acum
            best_params['n0'], best_params['iterations'] = n0, iterations
            best_params['c1'], best_params['c2'] = c1, c2
            best_params['P1'], best_params['P2'] = p1, p2
            best_params['N1'], best_params['N2'], best_params['N3'] = n1, n2, n3
    
        #print('--> #{} -- N0:{} iterations:{} c1:{} c2:{} -- mIoU-params : {} -- P1:{} P2:{} N1:{} N2:{} N3:{} \n\n'.format(i, n0, iterations, c1, c2, miou_test_acum, p1, p2, n1, n2, n3)
    
    df = pd.DataFrame(info_test)

    Excelwriter = pd.ExcelWriter("mIoU_data.xlsx",engine="xlsxwriter")
    df.to_excel(Excelwriter, sheet_name="mIoU",index=False)
    #for i, df in enumerate (info_searches_detailed):
    #    df.to_excel(Excelwriter, sheet_name=str(i),index=False)
    Excelwriter.save()

    #for i, df in enumerate (info_searches_detailed):
    #    df.to_csv('unit_tests/test_'+str(i)+'.csv', index=False)

    print('RANDOM SEARCH FINISHED!')
    print('Best result -- mIoU:',best_miou,'params:',best_params)
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation for FCANet")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'res2net'], help='backbone name (default: resnet)')
    parser.add_argument('--sis', action='store_true', default=False, help='use sis')

    parser.add_argument('--miou', type=float, default=-1.0, help='miou_target (default: -1.0[means automatic selection])')
    parser.add_argument('--dataset', type=str, default='VOC2012', help='evaluation dataset (default: VOC2012)')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu (not recommended)')
    args = parser.parse_args()

    print(datasets_path+'dataset/{}'.format(args.dataset))
    if Path(datasets_path+'dataset/{}'.format(args.dataset)).exists():
        model = None
        print("YAY!")
        findBestParams(args.dataset, 3, miou_target=(None if args.miou < 0 else args.miou))
        #eval_dataset(model,args.dataset,if_sis=args.sis, miou_target=(None if args.miou < 0 else args.miou),if_cuda=not args.cpu)
    
    """
    if Path('dataset/{}'.format(args.dataset)).exists():
        model = init_model('fcanet',args.backbone,'./pretrained_model/fcanet-{}.pth'.format(args.backbone),if_cuda=not args.cpu)
        eval_dataset(model,args.dataset,if_sis=args.sis, miou_target=(None if args.miou<0 else args.miou),if_cuda=not args.cpu)
    else:
        print('not found folder [dataset/{}]'.format(args.dataset))
    """