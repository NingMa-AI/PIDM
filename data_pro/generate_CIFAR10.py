import pickle as p
import numpy as np
import os
from PIL import Image

def load_CIFAR_batch(filename):
    """ 载入cifar数据集的一个batch """
    with open(filename, 'rb') as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ 载入cifar全部数据 """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)         #将所有batch整合起来
        ys.append(Y)
    Xtr = np.concatenate(xs) #使变成行向量,最终Xtr的尺寸为(50000,32,32,3)
    Ytr = np.concatenate(ys)
    del X, Y 
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def split_ssl_data( data, target, num_labels, num_classes, index=None, include_lb_to_ulb=True):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, = sample_labeled_data(data, target, num_labels, num_classes, index)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))  # unlabeled_data index of data
    if include_lb_to_ulb:
        return lb_data, lbs, data, target
    else:
        return lb_data, lbs, data[ulb_idx], target[ulb_idx]


def sample_labeled_data(data, target,
                        num_labels, num_classes,
                        index=None, name=None):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index

    # dump_path = os.path.join(args.save_dir, args.save_name, 'sampled_label_idx.npy')

    # if os.path.exists(dump_path):
    #     lb_idx = np.load(dump_path)
    #     lb_data = data[lb_idx]
    #     lbs = target[lb_idx]
    #     return lb_data, lbs, lb_idx

    samples_per_class = int(num_labels / num_classes)

    lb_data = []
    lbs = []
    lb_idx = []
    np.random.seed(2022)
    for c in range(num_classes):
        idx = np.where(target == c)[0]

        
   
        idx = np.random.choice(idx, samples_per_class, False)
        lb_idx.extend(idx)

        lb_data.extend(data[idx])
        lbs.extend(target[idx])

    # np.save(dump_path, np.array(lb_idx))
    # np.save(dump_path, np.array(lb_idx))

    return np.array(lb_data), np.array(lbs), np.array(lb_idx)



def mysave(dataset,lb_data,lbs, txt_path, ROOT,cnt):
    # cnt=0
    lines=[]
    with open(txt_path,"w") as f:
        isfirst=True
        for (img,label) in zip(lb_data,lbs):
            if not os.path.exists(os.path.join(ROOT,str(label))):
                os.makedirs(os.path.join(ROOT,str(label)))
            # print(type(img),img.shape)
            image=Image.fromarray(img.astype(np.uint8))
            image=image.convert('RGB')
            image.save(os.path.join(ROOT,str(label),"{}.jpg".format(cnt)))
            if isfirst:
                isfirst=False
                f.writelines(["{}/{}/{}.jpg {}".format(dataset,label,cnt,label)])
            else :
                f.writelines(["\n{}/{}/{}.jpg {}".format(dataset,label,cnt,label)])
            cnt+=1
        # f.writelines(lines)
    return cnt
ROOT="/data/maning/datasets/cifar-10-batches-py/"
Xtr, Ytr, Xte, Yte=load_CIFAR10(ROOT)
print(Xtr.shape,Ytr.shape,Xte.shape,Yte.shape)
num_classes=10
dataset="CIFAR10_4(1)"
num_labels=num_classes*4
lb_data, lbs, data, target=split_ssl_data( Xtr, Ytr, num_labels, num_classes, index=None, include_lb_to_ulb=True)

p=os.path.join(ROOT,"SSL",dataset)
import shutil

if os.path.exists(p):
    shutil.rmtree(p)

cnt=0
cnt1=mysave(dataset,lb_data,lbs,os.path.join("/data/maning/git/shot/data/SSDA_split/SSL","labeled_target_images_{}_{}.txt".format(dataset,int(num_labels/num_classes))),p,cnt)
cnt2=mysave(dataset,data,target,os.path.join("/data/maning/git/shot/data/SSDA_split/SSL","unlabeled_target_images_{}_{}.txt".format(dataset,int(num_labels/num_classes))),p,cnt1)
cnt3=mysave(dataset,Xte,Yte, os.path.join("/data/maning/git/shot/data/SSDA_split/SSL","validation_target_images_{}_{}.txt".format(dataset,int(num_labels/num_classes))),p,cnt2)
print(cnt1,cnt2,cnt3)


# np.save("/data/maning/datasets/cifar-10-batches-py/labeled_data.npy", np.array(lb_data))
# np.save("/data/maning/datasets/cifar-10-batches-py/labeled_label.npy", np.array(lbs))
# np.save("/data/maning/datasets/cifar-10-batches-py/unlabeled_data.npy", np.array(data))
# np.save("/data/maning/datasets/cifar-10-batches-py/unlabeled_label.npy", np.array(target))
# np.save("/data/maning/datasets/cifar-10-batches-py/test_data.npy", np.array(Xte))
# np.save("/data/maning/datasets/cifar-10-batches-py/test_label.npy", np.array(Yte))











