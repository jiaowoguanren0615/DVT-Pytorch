<h1 align='center'>Dendritic Vision Transformer (DVT)</h1>

# [DVT: Dendritic Learning-incorporated Vision Transformer for Image Recognition](https://www.ieee-jas.net/en/article/id/ec6a16fa-d348-417a-af0f-dd734c60439c)
This is a warehouse for DVT-Pytorch-model, can be used to train your image-datasets for classification tasks.  
The code mainly comes from [official source code](https://github.com/zhangzm0128/DVT/tree/main).  
___Now the code contain vit_base, vit_large, vit_huge models.___  

## Preparation
### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: DVT Model
    ├── build_models.py: Construct DVT models
    ├── SPT.py: Construct ShiftedPatchTokenization architecture
    ├── vit.py: Construct vision_transformer models
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── optimizer.py: Define Sophia optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  

## Use Sophia Optimizer (in util/optimizer.py)
You can use anther optimizer sophia, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
# optimizer = create_optimizer(args, model_without_ddp)
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=args.weight_decay)
```

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. If you recive an error like "___error: unrecognized arguments: --local-rank=1___" when you use distributed multi-GPUs training, just replace the command "___torch.distributed.launch___" to "___torch.distributed.run___".

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.launch --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{zhang2024dendritic,
  author={Zhiming Zhang,Zhenyu Lei,Masaaki Omura,Hideyuki Hasegawa,Shangce Gao},
  title={Dendritic Learning-Incorporated Vision Transformer for Image Recognition},
  journal={IEEE/CAA Journal of Automatica Sinica},  
  year={2024},
  volume={11},
  number={2},
  pages={541-543},
  doi={10.1109/JAS.2023.123978}
}
```
