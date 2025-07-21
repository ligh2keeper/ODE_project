from torch.utils.data.dataset import Dataset
import os
import torch
import pickle


class DiffeqDataset(Dataset):
    def __init__(self, train, batch_size, path):
        super(DiffeqDataset).__init__()
        self.train = train
        self.batch_size = batch_size
        self.path = path
        vocab_path = os.path.join(self.path, 'vocabulary')
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
            self.token2id = vocab['token2id']
            self.id2token = vocab['id2token']
            self.pad_idx = self.token2id['<pad>']

        data_path = os.path.join(self.path, 'data_train' if train else 'data_test')
        with open(data_path, 'r') as f:
            lines = [line.rstrip() for line in f]
            self.data = [eq_sol.split(',') for eq_sol in lines]
        self.size = len(self.data)

    def collate_fn(self, data):
        eqs, sols = zip(*data)
        eqs = [torch.LongTensor([int(token_id) for token_id in eq_sample]) for eq_sample in eqs]
        sols = [torch.LongTensor([int(token_id) for token_id in sol_sample]) for sol_sample in sols]
        eqs, eqs_len = self.form_batch(eqs)
        sols, sols_len = self.form_batch(sols)
        return eqs, sols

    def form_batch(self, seqs):
        lengths = torch.LongTensor([len(s) for s in seqs])
        batch_frame = torch.LongTensor(lengths.size(0), lengths.max().item()).fill_(self.pad_idx)
        #assert lengths.min().item() > 2
        for i, s in enumerate(seqs):
            batch_frame[i, :lengths[i]].copy_(s)
        return batch_frame, lengths

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.read_from_string_data(index)

    def read_from_string_data(self, index):
        eq, sol = self.data[index]
        eq_list = eq.split()
        sol_list = sol.split()
        return eq_list, sol_list
