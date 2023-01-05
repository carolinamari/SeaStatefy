#### Script de treino para CNN ####

# Imports
import os
from tqdm import tqdm
import time
import copy
import sys
import gc

import torchvision
from torchvision import datasets, models, transforms

from PIL import Image

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np


import matplotlib.pyplot as plt

cudnn.benchmark = True


#### Funções auxiliares ####

# Training
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    metrics = {}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # save metrics
            if phase == 'train':
                metrics[epoch] = {}
                metrics[epoch][phase] = {'acc': epoch_acc, 'loss': epoch_loss}

            else:
                metrics[epoch][phase] = {'acc': epoch_acc, 'loss': epoch_loss}

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

#### Main ####
def main():

    print('==== Inicializando processo ====')

    print('==== Analisando inputs ====')
    if len(sys.argv) != 5:
        print('ERRO: o comando deve conter estar no formato $ py training_script.py <record1>,<record2>,...,<recordN> <nome da rede> <learning rate> <nb epochs>')
        return
    
    datasets = sys.argv[1].split(',')
    network_name = sys.argv[2]
    learning_rate = float(sys.argv[3])
    nb_epochs = int(sys.argv[4])

    print('==== Carregando datasets ====')
    # Carrega as imagens pré-processadas de cada dataset
    base_dir = ''

    # with open(base_dir + 'YS_2017_2017-05-13_05-00-00_10Hz_2058_600.pkl', 'rb') as f:
    #     YS_2017 = pickle.load(f)

    # with open(base_dir + 'BS_2011_2011-10-04_11-38-00_12Hz_2058_600.pkl', 'rb') as f:
    #     BS_2011_1 = pickle.load(f)

    # with open(base_dir + 'LJ_2018_2018-01-03_09-39-38_10Hz_2058_600.pkl', 'rb') as f:
    #     LJ_2018 = pickle.load(f)

    print('==== Datasets caregados ====')

    print('==== Aplicando transformações ====')
    # Data augumentation para a classe 3
    transform_augumentation = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Transform to tensor
            transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
    ])
    # Normalização e conversão em tensores
    transform = transforms.Compose([
            transforms.ToTensor(),  # Transform to tensor
            transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
    ])

    # Concatena todos os sets em um e atribui os labels às imagens
    # em uma tupla: (<tensor>, <classe>)
    # data = [(transform(x), 0) for x in BS_2011_1]
    # data += [(transform(x), 1) for x in YS_2017]
    # data += [(transform_augumentation(x), 2) for x in LJ_2018]

    ### VERSÃO TESTE LEAVE OUT ###
    # datasets = ['AA_2014_2014-03-27_09-10-00_12Hz_2058_600.pkl', 'YS_2017_2017-05-13_05-00-00_10Hz_2058_600.pkl', 'BS_2011_2011-10-01_16-18-00_15Hz_2058_600.pkl',
    #         'BS_2011_2011-10-04_11-38-00_12Hz_2058_600.pkl', 'BS_2011_2011-10-04_13-07-00_12Hz_2058_600.pkl', 'BS_2013_2013-09-25_12-15-01_12Hz_2058_600.pkl',
    #         'LJ_2018_2018-01-03_09-39-38_10Hz_2058_600.pkl']
    # labels = [1, 1, 0, 0, 0, 0, 2]

    # datasets = ['AA_2014_2014-03-27_09-10-00_12Hz', 'YS_2017_2017-05-13_05-00-00_10Hz', 'BS_2011_2011-10-01_16-18-00_15Hz',
    #         'BS_2011_2011-10-04_11-38-00_12Hz', 'BS_2011_2011-10-04_13-07-00_12Hz', 'BS_2013_2013-09-25_12-15-01_12Hz',
    #         'LJ_2018_2018-01-03_09-39-38_10Hz']

    LABELS = {
        'AA_2014_2014-03-27_09-10-00_12Hz': 1,
        'AA_2015_2015-03-05_10-35-00_12Hz': 1,
        'AA_2015_2015-05-15_09-00-00_12Hz': 1,
        'YS_2017_2017-05-13_05-00-00_10Hz': 1,
        'BS_2011_2011-10-01_16-18-00_15Hz': 0,
        'BS_2011_2011-10-04_11-38-00_12Hz': 0,
        'BS_2011_2011-10-04_13-07-00_12Hz': 0,
        'BS_2011_2011-10-04_15-30-00_12Hz': 0,
        'BS_2013_2013-09-22_13-00-01_10Hz': 0,
        'BS_2013_2013-09-25_12-15-01_12Hz': 0,
        'BS_2013_2013-09-30_10-20-01_12Hz': 0,
        'LJ_2018_2018-01-03_09-39-38_10Hz': 2,
        'kaggle_1': 0,
        'kaggle_2': 0,
        'kaggle_3': 0,
        'kaggle_4': 1,
    }
    

    data = []
    for i in range(len(datasets)):
        with open(base_dir + datasets[i] + '_2058_224.pkl', 'rb') as f:
            record = pickle.load(f) 
        data += [(transform_augumentation(x), LABELS[datasets[i]]) for x in record]
        

        print(f'{datasets[i]}: OK')

    print('==== Transformações aplicadas ====')    

    
    # data = [(transform(img), label) for img, label in data]

    # Split data into training/test set (80/20)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])

    image_datasets = {'train': train_set, 'val': test_set}

    del data
    del train_set
    del test_set
    gc.collect()

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    
    # Class weightining
    arr_labels = [x[1] for x in image_datasets['train']]
    labels_unique, counts = np.unique(arr_labels, return_counts=True)
    print('Unique labels: {}'.format(labels_unique))
    class_weights = [sum(counts) / c for c in counts]

    ### Imbalance solved by oversampling ###
    # Assign weight to each input sample
    # sample_weights = [class_weights[e] for e in arr_labels]
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(arr_labels))

    # dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16,
    #                                             num_workers=4, sampler=sampler)}

    # dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=16,
    #                                             shuffle=True, num_workers=4)



    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = [1, 2, 3]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)


    print('==== Preparando o modelo para o treinamento ====')
    ### Resnet152 ###
    # # Carrega o modelo
    # resnet152 = models.resnet152(pretrained=True)
    # # resnet152 = models.resnet152(pretrained=False)

    # # Congela os parâmetros do feature extraction
    # for param in resnet152.parameters():
    #     param.requires_grad = False

    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = resnet152.fc.in_features
    # resnet152.fc = nn.Linear(num_ftrs, 3) # 3 classes

    # resnet152 = resnet152.to(device)

    # criterion = nn.CrossEntropyLoss()

    # ### Imbalace solved by clas weightining ###
    # # criterion = nn.CrossEntropyLoss(torch.FloatTensor(class_weights).to(device))

    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = optim.SGD(resnet152.fc.parameters(), lr=learning_rate, momentum=0.9)

    ### Resnet34 ###
    # Carrega o modelo
    # resnet34 = models.resnet34(pretrained=True)
    # # resnet34 = models.resnet34(pretrained=False)

    # # Congela os parâmetros do feature extraction
    # for param in resnet34.parameters():
    #     param.requires_grad = False

    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = resnet34.fc.in_features
    # resnet34.fc = nn.Linear(num_ftrs, 3) # 3 classes

    # resnet34 = resnet34.to(device)

    # criterion = nn.CrossEntropyLoss()

    # ### Imbalace solved by clas weightining ###
    # # criterion = nn.CrossEntropyLoss(torch.FloatTensor(class_weights).to(device))

    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = optim.SGD(resnet34.fc.parameters(), lr=learning_rate, momentum=0.9)

    ### VGG16 ###
    # Carrega o modelo
    # vgg16 = models.vgg16_bn(pretrained=True)
    # # vgg16 = models.vgg16_bn(pretrained=False)
    # # Freeze training for all layers
    # for param in vgg16.features.parameters():
    #     param.require_grad = False

    # # Newly created modules have require_grad=True by default
    # num_features = vgg16.classifier[6].in_features
    # features = list(vgg16.classifier.children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_features, 3)]) # Add our layer with 3 outputs
    # vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    # vgg16 = vgg16.to(device)

    # criterion = nn.CrossEntropyLoss()

    # ### Imbalace solved by clas weightining ###
    # # criterion = nn.CrossEntropyLoss(torch.FloatTensor(class_weights).to(device))

    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = optim.SGD(vgg16.classifier.parameters(), lr=learning_rate, momentum=0.9)

    ### Resnet34_VGG ###
    # Carrega o modelo
    resnet34_vgg = models.resnet34(pretrained=True)
    # resnet34 = models.resnet34(pretrained=False)

    # Congela os parâmetros do feature extraction
    for param in resnet34_vgg.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = resnet34_vgg.fc.in_features
    resnet34_vgg.fc = nn.Sequential(
        nn.Linear(in_features=num_ftrs, out_features=num_ftrs, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=3, bias=True) # 3 classes
    )

    resnet34_vgg = resnet34_vgg.to(device)

    # criterion = nn.CrossEntropyLoss()

    ### Imbalace solved by clas weightining ###
    criterion = nn.CrossEntropyLoss(torch.FloatTensor(class_weights).to(device))

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(resnet34_vgg.fc.parameters(), lr=learning_rate, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    print('==== Modelo preparado ====')

    print('==== Iniciando o treinamento ====')
    ### Resnet152 ###
    # model_conv, metrics = train_model(resnet152, criterion, optimizer_conv,
    #                      exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=nb_epochs)

    ### Resnet34 ###
    # model_conv, metrics = train_model(resnet34, criterion, optimizer_conv,
    #                      exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=nb_epochs)

    ### VGG16 ###
    # model_conv, metrics = train_model(vgg16, criterion, optimizer_conv,
    #                      exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=nb_epochs)

    ### Resnet34_VGG ###
    model_conv, metrics = train_model(resnet34_vgg, criterion, optimizer_conv,
                         exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=nb_epochs)

    print('==== Treinamento finalizado ====')

    print('==== Salvando modelo ====')
    torch.save(model_conv.state_dict(), f'{network_name}.pth')
    print('==== Modelo salvo ====')

    print('==== Salvando Metricas ====')
    with open(f'{network_name}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    print('==== Metricas salvas ====')

main()