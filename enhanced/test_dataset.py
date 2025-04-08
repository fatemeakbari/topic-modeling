from dataset import DocDataset
import pandas as pd
from models.GSM import GSM

if __name__ == '__main__':
    # Load your data
    df = pd.read_csv('./data/train.csv')
    texts = list(df.iloc[:, 1])
    labels = list(df.iloc[:, 0])

    # Create the dataset
    docSet = DocDataset(task_name="ag_news", texts=texts, labels=labels, no_below=5, no_above=0.0134, rebuild=False, use_tfidf=False)
    # checkpoint = torch.load('./ckpt/GSM_ag_news_tp10_2025-04-03-06-18_ep20.ckpt')
    # Create the DataLoader
    # data_loader = DataLoader(docSet, batch_size=128, shuffle=True, num_workers=4, collate_fn=docSet.collate_fn)

    # Iterate through the DataLoader
    model = GSM(bow_dim=docSet.vocab_size, n_topic=10, taskname="ag_news", device="cpu")
    model.train(train_data=docSet, batch_size=128, test_data=docSet, num_epochs=40, log_every=10, beta=1.0,
                criterion='cross_entropy')