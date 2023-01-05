import pickle

def main():
    with open('./224x224/models/resnet34_splitted_weighted_oversample_kaggle_tpn_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    print(metrics)

main()