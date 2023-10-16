# PIDM
Codes of "Source-free semi-supervised domain adaptation via progressive Mixup"
## Env setting
    Python=3.8
    Pytorh=1.7.0 (py3.8_cuda11.0.221)
    torchvision=0.8.1  

## Dataset setting
 1. All datasets can be seen in [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [office-home](https://www.hemanthdv.org/officeHomeDataset.html), [multi (DomainNet126)](http://csr.bu.edu/ftp/visda/2019/multi-source/). After these dataset are downloaded, please bulid a new folder named ./data/ and put the dataset into it.  An example is like:
    * DEEM-master/data
    
        * Office-31
        
            * webcam
        
                * images
                    * back_pack
                        * frame_0001.jpg
                        * frame_0002.jpg
                        * frame_0003.jpg
                    
               
2. To specify your dataset path, please set "project_root" in return_dataset.py

## Training usages
Training on DomainNet on 1-shot SSDA:
        python main.py --num 1 --dataset multi --s real --t clipart --gpu_id 0 --train 1
Test on DomainNet 1-shot SSDA:
        python main.py --num 1 --dataset multi --s real --t clipart --gpu_id 0 --train 0

The other datasets follow the similar usages!

If you find the repo is helpful, feel free to star and cite us:

        @article{MA2023110208,
        title = {Source-free semi-supervised domain adaptation via progressive Mixup},
        journal = {Knowledge-Based Systems},
        volume = {262},
        pages = {110208},
        year = {2023},
        issn = {0950-7051},
        doi = {https://doi.org/10.1016/j.knosys.2022.110208},
        url = {https://www.sciencedirect.com/science/article/pii/S0950705122013041},
        author = {Ning Ma and Haishuai Wang and Zhen Zhang and Sheng Zhou and Hongyang Chen and Jiajun Bu},
        keywords = {Domain adaptation, Semi-supervised learning, Data augmentation},
        abstract = {Existing domain adaptation methods usually perform explicit representation alignment by simultaneously accessing the source data and target data. However, the source data are not always available due to the privacy preserving consideration or bandwidth limitations. To address this issue, source-free domain adaptation is proposed to perform domain adaptation without accessing the source data. Recently, the adaptation paradigm is attracting increasing attention, and multiple works have been proposed for unsupervised source-free domain adaptation. However, without utilizing any supervised signal and source data at the adaptation stage, the optimization of the target model is unstable and fragile. To alleviate the problem, we focus on utilizing a few labeled target data to guide the adaptation, which forms our method into semi-supervised domain adaptation under a source-free setting. We propose a progressive data interpolation strategy including progressive anchor selection and dynamic interpolation rate to reduce the intra-domain discrepancy and inter-domain representation gap. Extensive experiments on three public datasets demonstrate the effectiveness as well as the better scalability of our method.}
        }
