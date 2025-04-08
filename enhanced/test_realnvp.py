from dataset import DocDataset
import pandas as pd
from models.GSM import GSM
import nltk
from torch.utils.data import DataLoader
import torch
# df = pd.read_csv('./data/train.csv')
# texts = list(df.iloc[:,1])
# labels = list(df.iloc[:,0])
# docSet = DocDataset(task_name="ag_news", texts=texts, labels=labels, no_below=5, no_above=0.0134, rebuild=False, use_tfidf=False)
#
# data_loader = DataLoader(docSet, batch_size=128, shuffle=True, num_workers=4,
#                          collate_fn=docSet.collate_fn)
#
# for iter,data in enumerate(data_loader):
#     pass
#
from realNVP import VAE
if __name__ == '__main__':
    # Load your data
    df = pd.read_csv('./data/train.csv')
    texts = list(df.iloc[:, 1])
    labels = list(df.iloc[:, 0])

    # Create the dataset
    docSet = DocDataset(task_name="ag_news", texts=texts, labels=labels, no_below=5, no_above=0.0134, rebuild=False, use_tfidf=False)
    checkpoint = torch.load('./ckpt/GSM_ag_news_tp10_2025-04-03-06-18_ep20.ckpt')
    # Create the DataLoader
    # data_loader = DataLoader(docSet, batch_size=128, shuffle=True, num_workers=4, collate_fn=docSet.collate_fn)

    # Iterate through the DataLoader
    model = GSM(bow_dim=docSet.vocab_size, n_topic=10, taskname="ag_news", device="cpu")
    vae = VAE("ag_new", 2, docSet.vocab_size, 512, 10, 0, None, 8).to('cpu')

    # model.train(train_data=docSet, batch_size=128, test_data=docSet, num_epochs=40, log_every=2, beta=1.0,
    #             criterion='cross_entropy', ckpt=checkpoint)
    data_loader = DataLoader(docSet, batch_size=128, shuffle=True, num_workers=4,
                             collate_fn=docSet.collate_fn)

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    trainloss_lst, valloss_lst = [], []
    recloss_lst, klloss_lst = [], []
    c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
    for epoch in range(0, 20):
        epochloss_lst = []
        for iter, data in enumerate(data_loader):
            optimizer.zero_grad()

            txts, bows = data
            bows = bows.to('cpu')
            bows_recon, mus, log_vars = vae(bows, lambda x: torch.softmax(x, dim=1))
            # if criterion == 'cross_entropy':
            logsoftmax = torch.log_softmax(bows_recon, dim=1)
            rec_loss = -1.0 * torch.sum(bows * logsoftmax)
            # elif criterion == 'bce_softmax':
            #     rec_loss = F.binary_cross_entropy(torch.softmax(bows_recon, dim=1), bows, reduction='sum')
            # elif criterion == 'bce_sigmoid':
            #     rec_loss = F.binary_cross_entropy(torch.sigmoid(bows_recon), bows, reduction='sum')

            kl_div = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

            loss = rec_loss + kl_div * 1.0

            loss.backward()
            optimizer.step()

            trainloss_lst.append(loss.item() / len(bows))
            epochloss_lst.append(loss.item() / len(bows))
            if (iter + 1) % 500 == 0:
                print(
                    f'Epoch {(epoch + 1):>3d}\tIter {(iter + 1):>4d}\tLoss:{loss.item() / len(bows):<.7f}\tRec Loss:{rec_loss.item() / len(bows):<.7f}\tKL Div:{kl_div.item() / len(bows):<.7f}')
        # scheduler.step()
        print(f'Epoch {(epoch + 1):>3d}\tLoss:{sum(epochloss_lst) / len(epochloss_lst):<.7f}')
        if (epoch + 1) % log_every == 0:
            if not os.path.exists('./ckpt'):
                os.mkdir('./ckpt')
            save_name = f'./ckpt/GSM_{self.taskname}_tp{self.n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_ep{epoch + 1}.ckpt'
            checkpoint = {
                "net": self.vae.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "param": {
                    "bow_dim": self.bow_dim,
                    "n_topic": self.n_topic,
                    "taskname": self.taskname
                }
            }
            torch.save(checkpoint, save_name)
            # The code lines between this and the next comment lines are duplicated with WLDA.py, consider to simpify them.
            print(f'Epoch {(epoch + 1):>3d}\tLoss:{sum(epochloss_lst) / len(epochloss_lst):<.7f}')
            print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            print('=' * 30)
            smth_pts = smooth_curve(trainloss_lst)
            plt.plot(np.array(range(len(smth_pts))) * log_every, smth_pts)
            plt.xlabel('epochs')
            plt.title('Train Loss')
            plt.savefig('gsm_trainloss.png')