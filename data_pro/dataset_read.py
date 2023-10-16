import sys

# sys.path.append('../data')
from data_pro.unaligned_data_loader import UnalignedDataLoader
from data_pro.svhn import load_svhn
from data_pro.mnist import load_mnist
from data_pro.mnist_m import load_mnistm
from data_pro.usps_ import load_usps
from data_pro.gtsrb import load_gtsrb
from data_pro.synth_number import load_syn
from data_pro.synth_traffic import load_syntraffic
 

def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist()
        #print(train_image.shape)
    if data == 'mnistm':
        train_image, train_label, \
        test_image, test_label = load_mnistm()
        #print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps()
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb()
    if data == 'syn':
        train_image, train_label, \
        test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label


def dataset_read(target, batch_size):
    S1 = {}
    S1_test = {}
    S2 = {}
    S2_test = {}
    S3 = {}
    S3_test = {}
    S4 = {}
    S4_test = {}
    
    S = [S1, S2, S3, S4]
    S_test = [S1_test, S2_test, S3_test, S4_test]

    T = {}
    T_test = {}
    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(target)

    target_train, target_train_label , target_test, target_test_label = return_dataset(target)
    print(domain_all)
    for i in range(len(domain_all)):
        source_train, source_train_label, source_test , source_test_label = return_dataset(domain_all[i])
        S[i]['imgs'] = source_train
        S[i]['labels'] = source_train_label
        #input target sample when test, source performance is not important
        S_test[i]['imgs'] = target_test
        S_test[i]['labels'] = target_test_label

    #S['imgs'] = train_source
    #S['labels'] = s_label_train
    T['imgs'] = target_train
    T['labels'] = target_train_label

    # input target samples for both 
    #S_test['imgs'] = test_target
    #S_test['labels'] = t_label_test
    T_test['imgs'] = target_test
    T_test['labels'] = target_test_label

    scale = 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()


    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    
    dataset_test = test_loader.load_data()

    return dataset, dataset_test
