import os
import torch
from torchvision import transforms
from data_pro.SSDA_data_list import Imagelists_VISDA, return_classlist
from data_pro.data_list import ImageList_idx,ImageList
import collections
import torch
import sys
import pickle
# project_root="."
import torch
import os
import random
import gensim # word2vec预训练加载
# # import jieba #分词
from torch import nn
import numpy as np
from numpy import *
# from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
# from zhconv import convert #简繁转换
import pickle
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
class CommentDataSet(Dataset):
    def __init__(self, data_text, word2idx, idx2word,memory_size,max_sentence_size):
        self.data_text = data_text
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.memory_size, self.max_sentence_size=memory_size,max_sentence_size
        self.data, self.label = self.get_data_label()
        self.get_index=0
    def __getitem__(self, idx: int):
        if self.get_index==1:
            # print("*")
            return self.data[idx], self.label[idx], idx
        else:

            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        data = []
        label = []
        for i,(review,domain,y) in enumerate(self.data_text):
            words_to_idx=[]
            for s_cnt, scetence in enumerate(review):
                if s_cnt==self.max_sentence_size: break
                for w_cnt, word in enumerate(scetence):
                    if w_cnt==self.memory_size: break
                    try:
                        index=self.word2idx[word]
                    except BaseException:
                        index=0
                    words_to_idx.append(index)
            label.append(torch.tensor(y, dtype=torch.int64))
            data.append(words_to_idx)
        # with open(self.data_path, 'r', encoding='UTF-8') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         try:
        #             label.append(torch.tensor(int(line[0]), dtype=torch.int64))
        #         except BaseException:  # 遇到首个字符不是标签的就跳过比如空行，并打印
        #             print('not expected line:' + line)
        #             continue
        #         # line = convert(line, 'zh-cn')  # 转换成大陆简体
        #         line_words = re.split(r'[\s]', line)[1:-1]  # 按照空字符\t\n 空格来切分
        #         words_to_idx = []
        #         for w in line_words:
        #             try:
        #                 index = self.word2idx[w]
        #             except BaseException:
        #                 index = 0  # 测试集，验证集中可能出现没有收录的词语，置为0
        #             #                 words_to_idx = [self.word2idx[w] for w in line_words]
        #             words_to_idx.append(index)
        #         data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label


def return_dataset(args):
    source_domain, target_domain=args.source_domain, args.target_domain

    def getVocab(data):
        """
        Get the frequency of each feature in the file named fname.
        """
        vocab = {}

        for review, _, _, in data:
            for sentence in review:
                for word in sentence:
                    vocab[word] = vocab.get(word, 0) + 1

        return vocab

    def get_review(f, domain, label):

        reviews = []
        y = 1  # sentiment label
        if label == "positive":
            y = 1
        elif label == "negative":
            y = 0

        with open(f, 'rb') as F:
            token_list = pickle.load(F)
            for tokens in token_list:
                # print tokens,"\n"
                reviews.append((tokens, domain, y))

        return reviews

    def load_bin_vec(fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            cnt = 0
            for line in range(vocab_size):
                cnt += 1
                print("line", cnt, line)
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                        # print("line", cnt, ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        return word_vecs

    def load_bin_vec_with_gensim(fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
        # print(word2vec_model.__dict__['vectors'].shape)
        word_vecs = {}

        for word, _ in word2vec_model.vocab.items():
            if word in vocab:
                # print(type(word2vec_model[word]))
                word_vecs[word] = np.array(word2vec_model[word], dtype='float32')

        return word_vecs

    def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)

    def get_w2vec(vocab, FLAGS):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        FLAGS.w2v_path = os.path.join(os.getcwd(), FLAGS.w2v_path)
        word_vecs = load_bin_vec_with_gensim(FLAGS.w2v_path, vocab)
        add_unknown_words(word_vecs, vocab)

        dim = len(list(word_vecs.values())[0])
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        idx_word_map = dict()
        W = np.zeros(shape=(vocab_size + 1, dim), dtype='float32')
        W[0] = np.zeros(dim, dtype='float32')
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            idx_word_map[i] = word
            i += 1

        return W, word_idx_map, idx_word_map

    def save_domain_word(file_path, emb, w2idx, idx2w):
        myword2vector = {}
        myword2vector["word_embedding"] = emb
        myword2vector["word2idx"] = w2idx
        myword2vector["idx2word"] = idx2w
        pickle.dump(myword2vector, open(file_path, "wb"))
        print("emb_saved")

    def load_domain_word(file_path):
        myword2vector = pickle.load(open(file_path, "rb"))
        print("emb_loaded")
        return myword2vector["word_embedding"], myword2vector["word2idx"], myword2vector["idx2word"]

    def mycollate_fn(data):
        # print(len(data))
        # 这里的data是getittem返回的（input，label）的二元组，总共有batch_size个
        data.sort(key=lambda x: len(x[0]), reverse=True)  # 根据input来排序
        data_length = [len(sq[0]) for sq in data]
        input_data = []
        label_data = []
        idex=[]
        for i in data:
            input_data.append(torch.from_numpy(np.array(i[0])))
            label_data.append(i[1])
            if len(data[0])==3:
                idex.append(i[2])
        input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
        label_data = torch.tensor(label_data)
        # idex=torch.tensor(idex)s
        if len(data[0]) == 3:
            # print("return idx")
            return input_data, label_data,idex, data_length
        else :

            return input_data, label_data, data_length

    def split_target(shot,unlabeled_target):
        labeled_target=[]
        test_data=[]
        val_target = []
        positive_cnt,negtive_cnt=0,0
        random.seed(0)
        indexes=list(range(len(unlabeled_target)))
        random.shuffle(indexes)
        # print("index",indexes)
        p_cnt,n_cnt=0,0

        for index in indexes:
            (review, domain, label)=unlabeled_target[index]
            if label==0 and negtive_cnt<shot:
                labeled_target.append((review,domain,label))
                negtive_cnt+=1
            elif label==1 and positive_cnt<shot:
                labeled_target.append((review,domain,label))
                positive_cnt+=1
            else:
                if random.random()<0.05:
                    val_target.append((review,domain,label))
                else:
                    test_data.append((review,domain,label))
            # if label==1:
            #           p_cnt+=1
            # else:v
            #     n_cnt+=1
        # print("split labeled, shot",shot,"labeled _len",len(labeled_target),"vallen",len(val_target),len(unlabeled_target),"test_len",len(test_data),p_cnt,n_cnt)
        return labeled_target,val_target,test_data

    train_data = []
    test_data = []
    val_data = []
    source_unlabeled_data = []
    target_unlabeled_data = []
    src, tar = 1, 0

    root_path=os.path.join(os.getcwd(),"data/han_amazon/")

    print("source domain: ", source_domain, "target domain:", target_domain)

    # load training data
    for (mode, label) in [("train", "positive"), ("train", "negative")]:
        fname = root_path+"%s/tokens_%s.%s" % (source_domain, mode, label)
        train_data.extend(get_review(fname, src, label))
    print ("train-size: ", len(train_data))

    # load validation data
    for (mode, label) in [("test", "positive"), ("test", "negative")]:
        fname = root_path+"/%s/tokens_%s.%s" % (source_domain, mode, label)
        val_data.extend(get_review(fname, src, label))
    print ("val-size: ", len(val_data))

    # load testing data
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("test", "positive"), ("test", "negative")]:
        fname = root_path+"%s/tokens_%s.%s" % (target_domain, mode, label)
        test_data.extend(get_review(fname, tar, label))
    print("test-size: ", len(test_data))

    # load unlabeled data
    for (mode, label) in [("train", "unlabeled")]:
        fname = root_path+"%s/tokens_%s.%s" % (source_domain, mode, label)
        source_unlabeled_data.extend(get_review(fname, src, label))
        fname = root_path+"%s/tokens_%s.%s" % (target_domain, mode, label)
        target_unlabeled_data.extend(get_review(fname, tar, label))
    print("unlabeled-size: ", len(source_unlabeled_data), len(target_unlabeled_data))

    vocab = getVocab(train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data)
    print ("vocab-size: ", len(vocab))

    # output_dir = "./work/logs/"
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    data = train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data
    source_data = train_data+val_data+source_unlabeled_data
    target_data = target_unlabeled_data

    max_story_size  = max(map(len, (pairs[0] for pairs in data)))
    mean_story_size = int(np.mean([len(pairs[0]) for pairs in data]))
    sentences = map(len, (sentence for pairs in data for sentence in pairs[0]))
    max_sentence_size = max(sentences)
    # mean_sentence_size = int(mean(sentences))
    memory_size = min(args.memory_size, max_story_size)
    # print("max  story size:", max_story_size)
    # print("mean story size:", mean_story_size)
    # print("max  sentence size:", max_sentence_size)
    # # print("mean sentence size:", mean_sentence_size)
    # print("max memory size:", memory_size)
    max_sentence_size = args.sent_size

    file_path = os.path.join(os.getcwd(), './data/han_amazon/', "w2vec_" + args.source_domain + "2" + args.target_domain + ".pkl")
    if os.path.exists(file_path):
        word_embedding, word2idx, idx2word=load_domain_word(file_path)
    else:
        word_embedding, word2idx, idx2word = get_w2vec(vocab, args)
        save_domain_word(file_path, word_embedding, word2idx, idx2word)

    train_data = CommentDataSet(train_data, word2idx, idx2word, memory_size, max_sentence_size)

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, collate_fn=mycollate_fn, )

    val_data = CommentDataSet(val_data, word2idx, idx2word, memory_size, max_sentence_size)

    validation_loader = DataLoader(val_data, batch_size=args.bs, shuffle=True,
                                   num_workers=args.workers, collate_fn=mycollate_fn, )

    labeled_target, val_target ,test_data= split_target(args.num, test_data)#sadfasdfasdfasdfasdfasdf

    target_test = CommentDataSet(test_data, word2idx, idx2word, memory_size, max_sentence_size)

    test_loader = DataLoader(target_test, batch_size=args.bs, shuffle=False,
                             num_workers=args.workers, collate_fn=mycollate_fn, )

    target_dataset_unl= CommentDataSet(test_data, word2idx, idx2word, memory_size, max_sentence_size)
    target_dataset_unl.get_index=1
    target_loader_unl=DataLoader(target_dataset_unl, batch_size=args.bs, shuffle=True,
                             num_workers=args.workers, collate_fn=mycollate_fn, )

    target_dataset_labeled = CommentDataSet(labeled_target, word2idx, idx2word, memory_size, max_sentence_size)

    target_loader = DataLoader(target_dataset_labeled, batch_size=args.bs, shuffle=True,
                                   num_workers=args.workers, collate_fn=mycollate_fn, )

    val_target_dataset = CommentDataSet(val_target, word2idx, idx2word, memory_size, max_sentence_size)

    target_loader_val = DataLoader(val_target_dataset, batch_size=args.bs, shuffle=True,
                                   num_workers=args.workers, collate_fn=mycollate_fn, )

    # return train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, vocab,word_embedding, word2idx, idx2word,memory_size,max_sentence_size
    return train_loader, validation_loader, target_loader, target_loader_unl, \
    target_loader_val, test_loader, word_embedding
    # return train_loader, validation_loader, test_loader,word_embedding,

def apply_train_dropout(m):
    if type(m) == torch.nn.Dropout:
        # print("find droput")
        m.train()

def return_dataloader_by_UPS(args,netF,netC,unlabeled_data_loader):
    netF.eval()
    netC.eval()
    base_path = project_root+"/data/SSDA_split/%s" % args.dataset

    if args.dataset in "office-home":
        # args.dataset='OfficeHomeDataset'
        root = project_root+"/data/OfficeHomeDataset/"
    else:
        root = project_root+"/data/%s/" % args.dataset

    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.t + '_3.txt')
    if args.uda==1:
        image_set_file_unl = \
            os.path.join(base_path,
                         'labeled_source_images_' +
                         args.t+".txt")
    else:
        image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    netF.apply(apply_train_dropout)# MC dropout
    netC.apply(apply_train_dropout)

    bs = args.batch_size
    unlabel_target_list = open(image_set_file_unl).readlines()
    target_list = open(image_set_file_t).readlines()

    target_dataset_unl = Imagelists_VISDA(unlabel_target_list, root=root,
                                          transform=TransformTwice(data_transforms['val']))
    target_dataset_unl.return_index=True
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                    batch_size=bs * 1, num_workers=args.worker,
                                                    shuffle=False, drop_last=False)
    start_test = True
    with torch.no_grad():
        all_index=[]
        for step, ((inputs1,inputs2), labels,index) in enumerate(target_loader_unl):
            # print(step,index)
             # = data[0]
            # labels = data[1]
            inputs1,inputs2 = inputs1.cuda(),inputs2.cuda()
            # labels = labels.cuda()  # 2020 07 06
            # inputs = inputs
            output=[]
            batch_s=inputs2.shape[0]
            repeat=5
            for i in range(repeat):
                outputs1 = netC(netF(inputs1)).cpu()
                outputs2 = netC(netF(inputs2)).cpu()
                output.append(outputs1)
                output.append(outputs2)
            output=torch.cat(output,dim=0).view(2*repeat,batch_s,-1)
            # print("std",torch.std(output,dim=0))

            outputs=torch.mean(output,dim=0)
            # print("outputs",outputs.shape)
            # print("mean",outputs)
            # outputs,margin_logits = netC(netF(inputs),labels)
            # labels = labels.cpu()  # 2020 07 06
            # step=torch.tensor(step).int().reshape(-1)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_index= index
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_index= torch.cat((all_index, index), 0)
            # print(type(index))
            # all_index.append(index)

    # print("len",len(all_index),len(all_output))
    _, pred_label = torch.max(all_output, 1)

    # _, predict = torch.max(all_output, 1)
    # res = torch.squeeze(predict).float() == all_label
    # accuracy = torch.sum(res).item() / float(all_label.size()[0])

    entropy = Entropy(torch.nn.Softmax(dim=1)(all_output))
    # mean_ent = torch.mean(entropy).cpu().data.item()

    # entropy for each class
    class_to_mean_entropy = list(0. for i in range(args.class_num))
    # correct = list(0. for i in range(args.class_num))
    total_add = list(0. for i in range(args.class_num))
    class2index=collections.defaultdict(list)
    classforentropy=collections.defaultdict(list)
    index2all_index=collections.defaultdict(dict)

    max_num_per_class=args.max_num_per_class
    # print(all_index.shape,pred_label.shape)
    for label in range(args.class_num):
        class_to_mean_entropy[label] = torch.mean(entropy[pred_label == label]).item()
        # classforentropy[label]=list(entropy[pred_label == label].numpy())

        index2all_index[label]=dict(list(zip(range(len(all_index[pred_label == label])),all_index[pred_label == label].numpy())))
        # print("index2all_label",index2all_index[label])
        classentropy=entropy[pred_label == label]
        if len(classentropy)<max_num_per_class:
            indexes=range(len(classentropy))
        else:
            preds, indexes = torch.topk(classentropy, max_num_per_class, largest=False)
            indexes=indexes.numpy()
        # print("entroy_len",len(entropy[pred_label == label]),"index_len",len(all_index[pred_label == label]))
        # print("indexes",indexes)
        for ind in indexes:
            class2index[label].append(index2all_index[label][ind]) # local index to global index


    total=0
    acc=0
    line2remove=[]

    if args.uda == 1:
        target_list=[]

    for psudo_label, indexes in class2index.items():
        for index in indexes:
            line= unlabel_target_list[index]
            psudo_line=line.split(" ")[0] + " " + str(psudo_label)
            target_list.append(psudo_line)
            line2remove.append(line)
            total_add[int(line.split(" ")[1])] += 1
            # if total_add[int(line.split(" ")[1])] >= max_num_per_class:
            #     break
            # print(line.split(" ")[1],str(psudo_label))
            if int(line.split(" ")[1])==psudo_label:
                acc+=1
            total+=1
    # print("acc of psudo label",acc*1.0/total)
    # print("added number for each class ",sum(total_add),total_add)
    before_remove=len(unlabel_target_list)
    for line in line2remove :
        unlabel_target_list.remove(line)
    # print("removed data from unlalbed dataset",before_remove," to ",len(unlabel_target_list))
    # print(target_list)


    target_dataset = Imagelists_VISDA(target_list, root=root,
                                      transform=data_transforms['val'])

    target_dataset_unl = Imagelists_VISDA(unlabel_target_list, root=root,
                                          transform=data_transforms['val'])

    target_dataset_unl.return_index=True

    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    # batch_size=bs,
                                    # pin_memory=True,
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    # pin_memory=True,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    netF.train()
    netC.train()
    return target_loader, target_loader_unl,acc*1.0/total

def return_dataset111111111111111111111(args):
    base_path = project_root + "/data/SSDA_split/%s" % args.dataset

    if args.dataset in "office-home":
        # args.dataset='OfficeHomeDataset'
        root = project_root + "/data/OfficeHomeDataset/"
    else:
        root = project_root + "/data/%s/" % args.dataset

    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.s + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.t + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def split(train_r, source_path):
        with open(source_path, 'r') as f:
            data = f.readlines()
            train_len = int(len(data) * train_r)
            train, val = torch.utils.data.random_split(data, [train_len, len(data) - train_len])
        return train, val

    if args.dataset in "multi":
        source_train, source_val = split(train_r=0.95, source_path=image_set_file_s)
    else:

        source_train, source_val = split(train_r=0.90, source_path=image_set_file_s)

    print("source_train and val num", len(source_train), len(source_val))

    source_dataset = Imagelists_VISDA(source_train, root=root,
                                      transform=data_transforms['train'])
    source_val_dataset = Imagelists_VISDA(source_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset = Imagelists_VISDA(open(image_set_file_t).readlines(), root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(open(image_set_file_t_val).readlines(), root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(open(image_set_file_unl).readlines(), root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl.return_index = True

    target_dataset_test = Imagelists_VISDA(open(image_set_file_unl).readlines(), root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    # if args.net == 'alexnet':
    #     bs = 20
    # else:
    #     bs = 16
    bs=args.batch_size
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=args.worker, shuffle=True,
                                                drop_last=False)
    source_val_loader = torch.utils.data.DataLoader(source_val_dataset, batch_size=bs,
                                                    num_workers=args.worker, shuffle=False,
                                                    drop_last=False)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=False)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=False, drop_last=False)
    return source_loader, source_val_loader, target_loader, target_loader_unl, \
           target_loader_val, target_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy