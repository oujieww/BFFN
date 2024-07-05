import math
import os
import random
import re
import seaborn as sns
import numpy as np
import matplotlib
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import pandas as pd
import torch
import torch.utils.data as Data
from torch import optim
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from modeling_bert import BertModel, BertForSequenceClassification,BertConfig
from modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertTokenizer
from transformers import AdamW
from peft import LoraConfig, TaskType, get_peft_model
from transformers.modeling_utils import ModuleUtilsMixin


model_path = './30virus_newMsame(roberta-wwm-ext)_doublelora0.25wd.pth'

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
best_acc = 0
best_test_acc = 0
train_txt_path = './30virus_train_newMsame(roberta-wwm-ext)att_doublelora0.25wd.txt'
test_txt_path = './30virus_test_newMsame(roberta-wwm-ext)att_doublelora0.25wd.txt'
eval_txt_path = './30virus_eval_newMsame(roberta-wwm-ext)att_doublelora0.25wd.txt'
cm_path = './30virus_newMsame(roberta-wwm-ext)att_doublelora0.25wd_cm'


def main():
    global epoch_num
    global best_epoch
    global best_acc
    global best_test_acc

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules= ["query","key","value","intermediate.dense"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False
    )


    model = BertForSequenceClassification.from_pretrained("./roberta-wwm-ext", num_labels=6)
    tokenizer_cn = BertTokenizer.from_pretrained("./roberta-wwm-ext")
    tokenizer_en = BertTokenizer.from_pretrained("./roberta-wwm-ext")

    model_en = get_peft_model(model, lora_config)
    model_cn = get_peft_model(model, lora_config)

    model = CustomModel(model_cn, model_en, num_labels=6)

    if torch.cuda.is_available():
        model.to(device)

    train_data = CustomDataset(pd.read_csv('./dataset/fusion_virus_train_xunfei.csv'), tokenizer_en, tokenizer_cn)
    eval_data = CustomDataset(pd.read_csv('./dataset/fusion_virus_eval_xunfei.csv'), tokenizer_en, tokenizer_cn)
    test_data = CustomDataset(pd.read_csv('./dataset/fusion_virus_test_xunfei.csv'), tokenizer_en, tokenizer_cn)


    for name, param in model.named_parameters():
        if "word_embeddings" in name:
            param.requires_grad=True

           param.requires_grad=True


    optimizer = optim.AdamW(model.parameters(), lr=2e-5)  # 优化器
    print_trainable_parameters(model)
    

    if not os.path.exists(cm_path):
        os.makedirs(cm_path)
        
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    epochs = 30  # 训练次数
    # 训练模型
    for i in range(epochs):
        epoch_num = i + 1       

        with open(train_txt_path, "a") as file:
            file.write("epoch: {}\n".format(epoch_num))

        with open(eval_txt_path, "a") as file:
            file.write("epoch: {}\n".format(epoch_num))
            
        with open(test_txt_path, "a") as file:
            file.write("epoch: {}\n".format(epoch_num))

        print("--------------- >>>> epoch : {} <<<< -----------------".format(epoch_num))
        train(model, train_data, criterion, optimizer)
        evaluate(model, eval_data, eval_txt_path, criterion)
        test(model, test_data, test_txt_path, criterion)
        
        if i == 0:
            for name, param in model.named_parameters():
                if "word_embeddings" in name:
                    param.requires_grad=False

    with open(eval_txt_path, "a") as file:
        file.write("best_acc: {}\n".format(best_acc))
        file.write("best_epoch: {}\n".format(best_epoch))

    with open(test_txt_path, "a") as file:
        file.write("best_acc: {}\n".format(best_acc))
        file.write("best_epoch: {}\n".format(best_epoch))
        file.write("best_test_acc: {}\n".format(best_test_acc))


    print("best_acc: {}".format(best_acc))
    print("best_epoch: {}".format(best_epoch))
    print("best_test_acc: {}\n".format(best_test_acc))


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved to {file_name}")

    
class CrossAttention(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert (
            self.head_dim * num_heads == embedding_dim
        ), "embedding_dim must be divisible by num_heads"

        self.query_audio = torch.nn.Linear(embedding_dim, embedding_dim)
        self.key_audio = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_audio = torch.nn.Linear(embedding_dim, embedding_dim)

        self.query_text = torch.nn.Linear(embedding_dim, embedding_dim)
        self.key_text = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_text = torch.nn.Linear(embedding_dim, embedding_dim)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.layer_norm_a = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm_l = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm_a1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm_l1 = torch.nn.LayerNorm(embedding_dim)

        self.feed_forward_a = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward_l = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, Ha, Hl):
        # Compute queries, keys, values
        Qa = self.query_audio(Ha)
        Ka = self.key_audio(Ha)
        Va = self.value_audio(Ha)

        Ql = self.query_text(Hl)
        Kl = self.key_text(Hl)
        Vl = self.value_text(Hl)

        # Calculate attention scores and apply softmax
        attention_scores_al = torch.matmul(Ql, Ka.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights_al = self.softmax(attention_scores_al)
        delta_Ha_l = torch.matmul(attention_weights_al, Va)

        attention_scores_la = torch.matmul(Qa, Kl.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
        attention_weights_la = self.softmax(attention_scores_la)
        delta_Hl_a = torch.matmul(attention_weights_la, Vl)

        # Update features with propagated information
        Ha_updated = self.layer_norm_a(Ha + delta_Hl_a)
        Hl_updated = self.layer_norm_l(Hl + delta_Ha_l)

        # Apply feed-forward layer
        Ha_final = self.layer_norm_a1(Ha_updated + F.relu(self.feed_forward_a(Ha_updated)))
        Hl_final = self.layer_norm_l1(Hl_updated + F.relu(self.feed_forward_l(Hl_updated)))

        return Ha_final, Hl_final    
    

class CustomDataset(Data.Dataset):
    def __init__(self, dataframe, tokenizer_en, tokenizer_cn, max_length=240):
        self.data = dataframe
        self.tokenizer_en = tokenizer_en
        self.tokenizer_cn = tokenizer_cn
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_en = self.data.iloc[idx, 0]  # English text
        text_cn = self.data.iloc[idx, 2]  # Chinese text
        label = self.data.iloc[idx, 1]  # Label
        # text_en = self.tokenize(text_en)
        # text_cn = self.tokenize(text_cn)

        # Tokenize English text
        encoding_en = self.tokenizer_en.encode_plus(
            text_en,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Tokenize Chinese text
        encoding_cn = self.tokenizer_cn.encode_plus(
            text_cn,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids_en': encoding_en['input_ids'].flatten(),
            'attention_mask_en': encoding_en['attention_mask'].flatten(),
            'token_type_ids_en': encoding_en['token_type_ids'].flatten(),
            'input_ids_cn': encoding_cn['input_ids'].flatten(),
            'attention_mask_cn': encoding_cn['attention_mask'].flatten(),
            'token_type_ids_cn': encoding_cn['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


#using cross-att
class CustomModel(torch.nn.Module):
    def __init__(self, model_en, model_cn, num_labels):
        super(CustomModel, self).__init__()
        self.bert_en = model_en.bert  # 使用内部的BERT模型
        self.bert_cn = model_cn.bert  # 使用内部的BERT模型

        # 交叉注意力层
        self.cross_attention = CrossAttention(embedding_dim=768, num_heads=12)     

        # 定义一个全连接层来将拼接后的向量压缩至6维，以匹配分类任务的需求
        # 注意这里的输入维度取决于你的拼接策略，这里假设每个表示都是768维
        self.fc = torch.nn.Linear(768, num_labels)  # 四倍的嵌入维度，因为我们拼接了四个表示

    def forward(self, input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn, token_type_ids_cn):
        # 处理英文输入
        outputs_en = self.bert_en(input_ids=input_ids_en, attention_mask=attention_mask_en, token_type_ids=token_type_ids_en)
        pooled_output_en = outputs_en.last_hidden_state[:, 0, :]

        # 处理中文输入
        outputs_cn = self.bert_cn(input_ids=input_ids_cn, attention_mask=attention_mask_cn, token_type_ids=token_type_ids_cn)
        pooled_output_cn = outputs_cn.last_hidden_state[:, 0, :]

        # 计算交叉注意力的输出
        Ha_final, Hl_final = self.cross_attention(outputs_en.last_hidden_state, outputs_cn.last_hidden_state)

  
        # 将BERT输出和交叉注意力输出进行拼接
        combined_output = (pooled_output_cn + pooled_output_en + Ha_final[:, 0, :] + Hl_final[:, 0, :]) / 4 
    
        # 通过全连接层将拼接后的向量压缩至6维
        logits = self.fc(combined_output)

        return logits


def calculate_metrics(true_labels, predicted_labels, filepath, mode):
    global cm_path
    num_classes = 6
    class_names = ['fear', 'happy', 'neutral', 'angry', 'surprise', 'sad']
    cm = confusion_matrix(true_labels, predicted_labels)

    cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.clf()  # 清空当前图形

    plt.figure(figsize=(10, 10))
    # plt.imshow(cm_prob, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
    plt.imshow(cm_prob, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{mode}-epoch{epoch_num}-ConfusionMatrix')

    threshold = cm_prob.max() / 2.
    cell_width = 1.0 / num_classes
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i - 0.1, format(cm[i, j], 'd'), ha='center', va='center',
                     color="white" if cm_prob[i, j] > threshold else "black", fontsize=14)
            plt.text(j, i + 0.2, format(cm_prob[i, j], '.4f'), ha='center', va='center', fontsize=14,
                     color="white" if cm_prob[i, j] > threshold else "black")

    plt.tight_layout()
    # plt.colorbar(ax=None)  # 不显示热力解释表

    output_image = f"{mode}_epoch{epoch_num}_confusionmatrix.png"
    output_path = os.path.join(cm_path, output_image)
    plt.savefig(output_path)
    plt.close()

    # 计算其他指标
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0.0)
    micro_recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0.0)
    micro_precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0.0)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0.0)
    macro_recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0.0)
    macro_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0.0)

    with open(filepath, "a") as file:
        file.write("Micro F1: {}\n".format(micro_f1))
        file.write("Micro Recall: {}\n".format(micro_recall))
        file.write("Micro Precision: {}\n".format(micro_precision))
        file.write("Macro F1: {}\n".format(macro_f1))
        file.write("Macro Recall: {}\n".format(macro_recall))
        file.write("Macro Precision: {}\n".format(macro_precision))

    print("Metrics data has been written to metrics.txt.")


def train(model, train_dataset, criterion, optimizer):
    loader_train = Data.DataLoader(dataset=train_dataset,
                                   batch_size=32,
                                   shuffle=False,
                                   collate_fn=collate_fn,
                                   drop_last=False)
    model.train()
    train_num = 0
    accuracy_num = 0
    epoch_loss = 0
    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储预测标签

    for i, (input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn, token_type_ids_cn,
            labels) in enumerate(loader_train):
        input_ids_en, attention_mask_en, token_type_ids_en = input_ids_en.to(device), attention_mask_en.to(
            device), token_type_ids_en.to(device)
        input_ids_cn, attention_mask_cn, token_type_ids_cn = input_ids_cn.to(device), attention_mask_cn.to(
            device), token_type_ids_cn.to(device)
        labels = labels.to(device)

        output = model(input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn,
                       token_type_ids_cn)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        
        
        num_embeddings_to_zero_en = int(0.75 * model.bert_en.embeddings.word_embeddings.weight.size(0))
        indices_to_zero_en = torch.randperm(model.bert_en.embeddings.word_embeddings.weight.size(0))[:num_embeddings_to_zero_en]
        # Set the gradient line for 75% of word embeddings to 0
        if model.bert_en.embeddings.word_embeddings.weight.grad is not None:
            model.bert_en.embeddings.word_embeddings.weight.grad[indices_to_zero_en] = 0

        num_embeddings_to_zero_cn = int(0.75 * model.bert_cn.embeddings.word_embeddings.weight.size(0))
        indices_to_zero_cn = torch.randperm(model.bert_cn.embeddings.word_embeddings.weight.size(0))[:num_embeddings_to_zero_cn]
        # Set the gradient line for 75% of word embeddings to 0
        if model.bert_cn.embeddings.word_embeddings.weight.grad is not None:
            model.bert_cn.embeddings.word_embeddings.weight.grad[indices_to_zero_cn] = 0    
            
        optimizer.step()

        output = output.argmax(dim=1)  # 取出所有在维度 1 上的最大值的下标
        accuracy_num += (output == labels).sum().item()
        train_num += len(input_ids_en)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(output.tolist())
        epoch_loss += loss.item()

    true_labels = torch.tensor(true_labels)
    predicted_labels = torch.tensor(predicted_labels)
    epoch_loss /= len(loader_train)
    total_train_acc = accuracy_num / train_num
    print("total train_loss: {}".format(epoch_loss))
    print("total train_acc: {}".format(total_train_acc))

    with open(train_txt_path, "a") as file:
        file.write("loss: {}\n".format(epoch_loss))
        file.write("Accuracy: {}\n".format(total_train_acc))

    calculate_metrics(true_labels, predicted_labels, train_txt_path, "train")


def test(model, test_dataset, filename, criterion):
    global epoch_num
    global best_epoch
    global best_test_acc
    correct_num = 0
    test_num = 0
    epoch_loss = 0
    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储预测标签
    global best_acc
    loader_test = Data.DataLoader(dataset=test_dataset,
                                  batch_size=32,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  drop_last=False)
    model.eval()

    for i, (input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn, token_type_ids_cn,
            labels) in enumerate(loader_test):
        input_ids_en, attention_mask_en, token_type_ids_en = input_ids_en.to(device), attention_mask_en.to(
            device), token_type_ids_en.to(device)
        input_ids_cn, attention_mask_cn, token_type_ids_cn = input_ids_cn.to(device), attention_mask_cn.to(
            device), token_type_ids_cn.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn,
                           token_type_ids_cn)

        loss = criterion(output, labels)
        output = output.argmax(dim=1)
        correct_num += (output == labels).sum().item()
        test_num += len(input_ids_en)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(output.tolist())
        # if t % 10 == 0:
        # print("schedule: [{}/{}] acc: {}".format(t, len(loader_test), correct_num / test_num))
        epoch_loss += loss.item()

    epoch_loss /= len(loader_test)
    print("total test_loss: {}".format(epoch_loss))

    totaltest_acc = correct_num / test_num
    print("total test_acc: {}".format(totaltest_acc))

    with open(filename, "a") as file:
        file.write("loss: {}\n".format(epoch_loss))
        file.write("Accuracy: {}\n".format(totaltest_acc))

    true_labels = torch.tensor(true_labels)
    predicted_labels = torch.tensor(predicted_labels)
    calculate_metrics(true_labels, predicted_labels, filename, "test")

    if totaltest_acc > best_test_acc:
        best_test_acc = totaltest_acc
        save_model(model, model_path)
     
    if totaltest_acc > best_acc:
        best_epoch = epoch_num
        best_acc = totaltest_acc
        print("test_best_acc: {}".format(totaltest_acc))
        

def evaluate(model, test_dataset, filename, criterion):
    global epoch_num
    global best_epoch
    correct_num = 0
    test_num = 0
    epoch_loss = 0
    true_labels = []  # 存储真实标签
    predicted_labels = []  # 存储预测标签
    global best_acc
    loader_test = Data.DataLoader(dataset=test_dataset,
                                  batch_size=32,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  drop_last=False)
    model.eval()

    for i, (input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn, token_type_ids_cn,
            labels) in enumerate(loader_test):
        input_ids_en, attention_mask_en, token_type_ids_en = input_ids_en.to(device), attention_mask_en.to(
            device), token_type_ids_en.to(device)
        input_ids_cn, attention_mask_cn, token_type_ids_cn = input_ids_cn.to(device), attention_mask_cn.to(
            device), token_type_ids_cn.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn,
                           token_type_ids_cn)

        loss = criterion(output, labels)
        output = output.argmax(dim=1)
        correct_num += (output == labels).sum().item()
        test_num += len(input_ids_en)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(output.tolist())
        # if t % 10 == 0:
        # print("schedule: [{}/{}] acc: {}".format(t, len(loader_test), correct_num / test_num))
        epoch_loss += loss.item()

    epoch_loss /= len(loader_test)
    print("total eval_loss: {}".format(epoch_loss))

    totaltest_acc = correct_num / test_num
    print("total eval_acc: {}".format(totaltest_acc))

    with open(filename, "a") as file:
        file.write("loss: {}\n".format(epoch_loss))
        file.write("Accuracy: {}\n".format(totaltest_acc))

    true_labels = torch.tensor(true_labels)
    predicted_labels = torch.tensor(predicted_labels)
    calculate_metrics(true_labels, predicted_labels, filename, "eval")

#     if totaltest_acc > best_acc:
#         best_epoch = epoch_num
#         best_acc = totaltest_acc
#         print("eval_best_acc: {}".format(totaltest_acc))


def collate_fn(batch):
    input_ids_en = torch.stack([item['input_ids_en'] for item in batch])
    attention_mask_en = torch.stack([item['attention_mask_en'] for item in batch])
    token_type_ids_en = torch.stack([item['token_type_ids_en'] for item in batch])
    input_ids_cn = torch.stack([item['input_ids_cn'] for item in batch])
    attention_mask_cn = torch.stack([item['attention_mask_cn'] for item in batch])
    token_type_ids_cn = torch.stack([item['token_type_ids_cn'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return input_ids_en, attention_mask_en, token_type_ids_en, input_ids_cn, attention_mask_cn, token_type_ids_cn, labels


if __name__ == '__main__':
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 全局变量
    print('所用的设备为(cuda即为gpu): ', device)
    main()
