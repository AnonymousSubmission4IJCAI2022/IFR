# coding: UTF-8
import time, os
import torch
import numpy as np
import argparse
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.model_selection import KFold
from tqdm import tqdm
import time
import logging
import random 


PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding labels

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim= -1)

    def forward(self, x):
        context = x[0] 
        mask = x[2]  
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        out = self.softmax(out)
        return out


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


class Config(object):
    def __init__(self, args):
        self.base_path = args.base_path
        print('Base path: {}'.format(self.base_path))
        self.train_path = os.path.join(args.base_path, 'inputs', args.train_file)
        print('Training file path: {}'.format(self.train_path))
        self.test_path = ''
        if len(args.test_file) != 0:
            self.test_path = os.path.join(args.base_path, 'inputs', args.test_file)
        print('Testing file path: {}'.format(self.test_path)) 
        self.inaccurate_path = ''
        if len(args.inaccurate_file) != 0:
            self.inaccurate_path = os.path.join(args.base_path, 'inputs', args.inaccurate_file)
        print('Inaccurate file path: {}'.format(self.inaccurate_path)) 
        self.class_list = [i.strip() for i in args.class_list.split(',')]
        self.str2int_map, self.int2str_map = self.class_map_func()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.class_list)
        print('Classes: {}'.format(self.num_classes))
        self.num_epochs = args.epoch
        print('Epochs: {}'.format(self.num_epochs))
        self.batch_size = args.batch
        print('Batch size: {}'.format(self.batch_size))
        self.pad_size = args.pad
        print('Pad size: {}'.format(self.pad_size))
        self.learning_rate = 5e-5
        print('Learning rate: {}'.format(self.learning_rate))
        self.bert_path = args.bert
        self.lang = args.lang
        if self.lang == 'cn':
            self.bert_path = './bert_pretrain'
        print('Pre-trained BERT path: {}'.format(self.bert_path))
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        print('Hidden size: {}'.format(self.hidden_size))
        self.txt_a_idx = [int(item) for item in args.txt_a.split(',')]
        print('Text A idx: {}'.format(self.txt_a_idx))
        if len(args.txt_b) > 0:
            self.txt_b_idx = [int(item) for item in args.txt_b.split(',')]
        else:
            self.txt_b_idx = ''
        print('Text B idx: {}'.format(self.txt_b_idx))
        self.label_idx = args.label_idx
        print('Label idx: {}'.format(self.label_idx))
        self.random_seed = args.random_seed
        print('Random seed: {}'.format(self.random_seed))
        self.separator = args.separator
        print('Separator: {}'.format(self.separator))
        self.cv_nums = args.cv_nums
        print('Cross-validation num: {}'.format(self.cv_nums))
        self.choice_rate = args.choice_rate
        print('Choice rate: {}'.format(self.choice_rate))
        self.choice_rate_train = args.choice_rate_train
        print('Choice rate for training: {}'.format(self.choice_rate_train))
        self.choice_num = args.choice_num
        print('Choice num: {}'.format(self.choice_num))
        self.iteration_threshold = args.iteration_threshold
        print('Iteration threshold: {}'.format(self.iteration_threshold))
        self.job_name = args.job_name
        print('Job name: {}'.format(self.job_name))
        self.model_path = os.path.join(args.base_path, 'models', str(self.job_name) + '_' +  str(self.num_epochs) + args.train_file.split('/')[-1] + '.ckpt')
        print('Model path: {}'.format(self.model_path))
        self.log_path = os.path.join(args.base_path, 'ira_logs', str(self.job_name) + '_' +  str(self.num_epochs) + args.train_file.split('/')[-1] + '.log')
        print('Logging path: {}'.format(self.log_path))

    def class_map_func(self):
        dict_, reverse_dict_ = {}, {}
        for i in range(len(self.class_list)):
            dict_[self.class_list[i].strip()] = i
            reverse_dict_[i] = self.class_list[i].strip()
        print(dict_)
        print(reverse_dict_)
        return dict_, reverse_dict_


def init_network(model, method='xavier', exclude='embedding', seed=1024):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def build_dataset(config):
    def load_dataset(path, pad_size):
        contents = []
        labels = []
        involved_items = []
        if len(path) == 0:
            return contents, labels, involved_items
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                tmp_list = lin.split(config.separator)
                if len(tmp_list) < config.label_idx or len(tmp_list) < config.txt_a_idx[-1] or len(str(tmp_list[config.label_idx])) == 0 or (str(tmp_list[config.label_idx]) not in config.class_list):
                    continue
                txt_a = ''
                for i in range(len(config.txt_a_idx)):
                    txt_a += tmp_list[config.txt_a_idx[i]]
                txt_b = ''
                if len(config.txt_b_idx) > 0:
                    for i in range(len(config.txt_b_idx)):
                        txt_b += tmp_list[config.txt_b_idx[i]]
                label = int(config.str2int_map[str(tmp_list[config.label_idx])])
                labels.append(label)
                involved_items.append(tmp_list)
                token_a = config.tokenizer.tokenize(txt_a)
                if len(txt_b) > 0:
                    token_b = config.tokenizer.tokenize(txt_b)
                    token = [CLS] + token_a + [SEP] + token_b + [SEP]
                else:
                    token = [CLS] + token_a + [SEP] 
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents, labels, involved_items
    train, train_labels, _ = load_dataset(config.train_path, config.pad_size)
    test, test_labels, _ = load_dataset(config.test_path, config.pad_size)
    inaccurate, inaccurate_labels, _ = load_dataset(config.inaccurate_path, config.pad_size)
    return train, train_labels, test, test_labels, inaccurate, inaccurate_labels


def train_cv(config, model, train_data, test_iter, acc_t):
    prev_model = model
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_data) * config.num_epochs)
    kf = KFold(n_splits = config.cv_nums)
    for train_index, val_index in kf.split(train_data):
        train_iter = DatasetIterater(train_data[train_index.astype(int)], config.batch_size, config.device)
        dev_iter = DatasetIterater(train_data[val_index.astype(int)], config.batch_size, config.device)
        model.train()
        for epoch in range(config.num_epochs):
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                dev_acc, dev_loss, _, _, _ = evaluate(config, model, dev_iter)
                model.train()
                msg = 'epoch: {0: >6}, batch: {1: >6}, val acc: {2: >6.4}, val loss: {3: >6.4}'
                # logging.info(msg.format(epoch + 1, i + 1,  dev_acc, dev_loss))
    ret_list, labels_one_hot, logits_one_hot = [], [], []
    if len(config.test_path) != 0:
        test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot = test_4cv(config, model, test_iter, False)
        if acc_t > 0 and test_acc < acc_t:
            model = prev_model
            test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot = test_4cv(config, model, test_iter, False)
        else:
            torch.save(model.state_dict(), config.model_path)
    return test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot, model

def train_no_cv(config, model, train_data, test_iter, iteration):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_data) * config.num_epochs)
    train_iter = DatasetIterater(train_data, config.batch_size, config.device)
    model.train()
    for epoch in range(config.num_epochs):
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # logging.info('w/ cv iteration: {0: >6}, epoch: {1: >6}, batch: {2: > 6}'.format(iteration + 1, epoch + 1, i + 1))
    ret_list, labels_one_hot, logits_one_hot = [], [], []
    if len(config.test_path) != 0:
        test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot = test(config, model, test_iter, False)
    return test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot


def test_4cv(config, model, test_iter, single = True):
    if single == True:
        model.load_state_dict(torch.load(config.model_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion, ret_list, labels_one_hot, logits_one_hot = evaluate(config, model, test_iter, True)
    logging.info('test loss: {0: >5.2}, test acc: {1: >6.2%}'.format(test_loss, test_acc))
    logging.info("Precision, Recall and F1-Score: ")
    logging.info(test_report)
    logging.info("Confusion Matrix: ")
    logging.info(test_confusion)
    return test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot


def test(config, model, test_iter, single = True):
    if single == True:
        model.load_state_dict(torch.load(config.model_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion, ret_list, labels_one_hot, logits_one_hot = evaluate(config, model, test_iter, True)
    # logging.info('test loss: {0: >5.2}, test acc: {1: >6.2%}'.format(test_loss, test_acc))
    # logging.info("Precision, Recall and F1-Score: ")
    # logging.info(test_report)
    # logging.info("Confusion Matrix: ")
    # logging.info(test_confusion)
    return test_acc, test_loss, ret_list, labels_one_hot, logits_one_hot

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    ret_list = []
    labels_one_hot = []
    logits_one_hot = []
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            ret_ = outputs.cpu().numpy()
            for item in ret_:
                ret_list.append(item)
            labels_one_hot.extend(labels.cpu().detach().numpy().tolist())
            logits_one_hot.extend(outputs.cpu().detach().numpy().tolist())
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    tmp_label = np.array(labels_one_hot)
    labels_one_hot = np.zeros((tmp_label.size, config.num_classes))
    labels_one_hot[np.arange(tmp_label.size), tmp_label] = 1
    if test == True:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, ret_list, labels_one_hot, np.asarray(logits_one_hot)
    return acc, loss_total / len(data_iter), ret_list, labels_one_hot, np.array(logits_one_hot)


def test_inaccurate(config, model, inaccurate_iter, single = True):
    if len(config.inaccurate_path) == 0:
        return 
    if single == True:
        model.load_state_dict(torch.load(config.model_path))
    model.eval()
    ret_list = []
    labels_ = []
    logits = []
    with torch.no_grad():
        for texts, labels in inaccurate_iter:
            outputs = model(texts)
            ret_ = outputs.cpu().numpy()
            for item in ret_:
                ret_list.append(item)
            labels_.extend(labels.cpu().detach().numpy().tolist())
            logits.extend(outputs.cpu().detach().numpy().tolist())
    return np.asarray(ret_), np.asarray(labels_), np.asarray(logits)


def confidence_matrix(logis, inaccurate_data, inaccurate_true_labels):
    filtered_subset = []
    filtered_subset_labels = []
    threshold_list = np.mean(logis, axis=0)
    def have_one_bigger(threshold_list, logis_one, label_one):
        for i in range(len(threshold_list)):
            if logis_one[i] >= threshold_list[i] and int(label_one) != int(i):
                return False
        return True
    for i in range(len(inaccurate_data)):
        if have_one_bigger(threshold_list, logis[i], inaccurate_true_labels[i]):
            filtered_subset.append(inaccurate_data[i])
            filtered_subset_labels.append(inaccurate_true_labels[i])
    return filtered_subset, filtered_subset_labels


def random_choice(config, filtered_subset, filtered_subset_labels):
    subset_idx = [i for i in range(len(filtered_subset_labels))]
    choice_nums = int(len(filtered_subset_labels) * float(config.choice_rate))
    filltered_idx = random.sample(subset_idx, k = choice_nums)
    chose_data = []
    chose_labels = []
    for i in filltered_idx:
        chose_data.append(filtered_subset[i])
        chose_labels.append(filtered_subset_labels[i])
    return filltered_idx, chose_data, chose_labels


def penalty(filtered_subset):
    return [[i, 1.0] for i in range(len(filtered_subset))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFR')
    parser.add_argument('--class_list', type=str, help='the class list seperated by ,', default='0, 1')
    parser.add_argument('--batch', type=int, help='batch size', default=256)
    parser.add_argument('--epoch', type=int, help='epoch number', default=5)
    parser.add_argument('--base_path', type=str, help='data dir', default='./data/')
    parser.add_argument('--train_file', type=str, required=True, help='train data file')
    parser.add_argument('--test_file', type=str, required=False, help='test data file')
    parser.add_argument('--inaccurate_file', type=str, required=False, help='inaccurate data file')
    parser.add_argument('--pad', type=int, help='pad size', default=64)
    parser.add_argument('--bert', type=str, help='pertrained bert path', default='./bert_pretrain_en')
    parser.add_argument('--txt_a', type=str, help='the index of txt_a', required=True, default="0")
    parser.add_argument('--txt_b', type=str, help='the index of txt_b', default="")
    parser.add_argument('--label_idx', type=int, help='the index of label', default=1)
    parser.add_argument('--random_seed', type=int, help='random seed', default=1024)
    parser.add_argument('--separator', type=str, help='separator for text A and B', default='\t')
    parser.add_argument('--cv_nums', type=int, help='cross validation number', default=5)
    parser.add_argument('--job_name', type=str, help='job name', default='')
    parser.add_argument('--lang', type=str, default='en', help='choose the language of the data.')    
    parser.add_argument('--choice_rate', type=float, help='choice rate of inaccurate data', default=0.3)
    parser.add_argument('--choice_num', type=int, help='choice number of inaccurate data', default=13)
    parser.add_argument('--iteration_threshold', type=int, help='iteration_threshold of training inaccurate data', default=10) 
    parser.add_argument('--choice_rate_train', type=float, help='choice rate', default=0.1)
    args = parser.parse_args()

    config = Config(args)
    logging.basicConfig(filename = config.log_path, level = logging.DEBUG)
   
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    train_data, train_true_label, test_data, test_true_label, inaccurate_data, inaccurate_true_labels = build_dataset(config)
    test_iter = []
    if len(test_data) != 0:
        test_iter = DatasetIterater(test_data, config.batch_size, config.device)
    time_diff = time.time() - start_time
    logging.info("Loading data time usage: {}".format(time_diff))

    model = Model(config).to(config.device)
    if torch.cuda.device_count() > 1:
        model_start = model = nn.DataParallel(model)
    acc_t = 0
    print("Original")
    acc_t, _, _, _, _, model = train_cv(config, model, np.array(train_data), test_iter, acc_t)
    acc_origin = 0
    iteration = 0
    while(iteration < config.iteration_threshold):
        iteration += 1
        acc_origin = acc_t
        inaccurate_iter = []
        if len(inaccurate_data) != 0:
            inaccurate_iter = DatasetIterater(inaccurate_data, config.batch_size, config.device)
        else:
            exit()
        _, _, logis = test_inaccurate(config, model, inaccurate_iter, False)
        filtered_subset, filtered_subset_labels = confidence_matrix(logis, inaccurate_data, inaccurate_true_labels)
        filtered_subset_penalty = penalty(filtered_subset)
        for i in range(int(config.choice_num)):
            chose_model = Model(config).to(config.device)
            filtered_idx, chose_data, chose_labels = random_choice(config, filtered_subset, filtered_subset_labels)
            chose_acc, _, _, _, _ = train_no_cv(config, chose_model, np.array(chose_data), test_iter, iteration)
            for j in filtered_idx:
                filtered_subset_penalty[j][1] *= float(chose_acc)
        filtered_subset_penalty_idx_filtered = sorted(filtered_subset_penalty, key = (lambda x: x[1]), reverse = True)[:int(config.choice_rate_train * len(filtered_subset_penalty))]
        print('filtered subset size: %d'%(len(filtered_subset_penalty_idx_filtered)))
        sampling_set = []
        sampling_set_labels = []
        for item in filtered_subset_penalty_idx_filtered:
            sampling_set.append(filtered_subset[item[0]])
            sampling_set_labels.append(filtered_subset_labels[item[0]])
        print("iteration : {}".format(iteration))
        acc_t, _, _, _, _, model = train_cv(config, model, np.array(sampling_set), test_iter, acc_t)
    print('Test acc: {0: >6.4%}'.format(acc_t))
    print('Model saved at: {}'.format(config.model_path))


