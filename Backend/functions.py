from torchvision import models, transforms

import torch
import torch.nn as nn

import numpy as np

def classify_img(img):
    ### Resnet34
    model = models.resnet34()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    min_dim = min(list(img.size))

    # Transformações a serem aplicadas nas imagens
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop(min_dim),
        transforms.Resize(224),
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
    ])

    # Aplicando as transformações na imagem
    img_tensor = transform(img)

    # Carrega o modelo
    model_name = 'resnet34_splitted_weighted_oversample_kaggle_tpn'
    model.load_state_dict(torch.load(f'./{model_name}.pth', map_location=lambda storage, loc: storage))

    dataloaders = {'val': torch.utils.data.DataLoader([img_tensor], batch_size=1,
                                             shuffle=True, num_workers=4)}

    # Modo de teste
    model.eval()

    with torch.no_grad():
        for i, inputs in enumerate(dataloaders['val']):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Conversão em classe e probabilidades
            classe = int(preds[0])
            probs = np.array(nn.functional.softmax(outputs)[0])*100

            return classe, probs


def set_response(classe, probs):
    # Dicionario de respostas
    responses = {
        0: {
            'classe_id': 1,
            'classe': '0 - 3',
            'descricao': 'Calmo a brisa fraca',
            'velocidade_do_vento': '0 - 5,4 m/s',
            'altura_media_ondas': '0 - 0,6 m',
            'aspecto_mar': {
                'from': 'Mar espelhado',
                'to': 'Ondulações com cristas que ocasionalmente possuem espuma; rebentações mais frequentes'
            },
            'probabilidades': [
                {
                    'range': 'Classes 0 - 3',
                    'p': 0
                },
                {
                    'range': 'Classes 4 - 7',
                    'p': 0
                },
                {
                    'range': 'Classes 8 - 12',
                    'p': 0
                }
            ]
        },
        1: {
            'classe_id': 2,
            'classe': '4 - 7',
            'descricao': 'Brisa moderada a vento forte',
            'velocidade_do_vento': '5,5 - 17,1 m/s',
            'altura_media_ondas': '1 - 4 m',
            'aspecto_mar': {
                'from': 'Pequenas ondas mais longas com cristas de espuma',
                'to': 'Mar revolto com espuma e borrifos soprados na direção dos ventos'
            },
            'probabilidades': [
                {
                    'range': 'Classes 0 - 3',
                    'p': 0
                },
                {
                    'range': 'Classes 4 - 7',
                    'p': 0
                },
                {
                    'range': 'Classes 8 - 12',
                    'p': 0
                }
            ]
        },
        2: {
            'classe_id': 3,
            'classe': '8 - 12',
            'descricao': 'Ventania a furacão',
            'velocidade_do_vento': '17,2 - 32,7 m/s',
            'altura_media_ondas': '5,5 - 14 m',
            'aspecto_mar': {
                'from': 'Mar revolto com ondas moderadamente altas, rebentações e borrifos constantes; formação de faixas de espumas bem marcadas ao longo da direção do vento',
                'to': 'Mar e ar preenchidos por espuma e borrifos; mar branco; visibilidade quase nula'
            },
            'probabilidades': [
                {
                    'range': 'Classes 0 - 3',
                    'p': 0
                },
                {
                    'range': 'Classes 4 - 7',
                    'p': 0
                },
                {
                    'range': 'Classes 8 - 12',
                    'p': 0
                }
            ]
        }
    }

    response = responses[classe]
    response['probabilidades'] = []
    for i, x in enumerate(probs):
        classe_range = responses[i]['classe']
        response['probabilidades'].append({'range': f'Classes {classe_range}', 'p': float(x)})

    return response