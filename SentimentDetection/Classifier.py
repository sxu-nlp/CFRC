import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import argparse
import torch.optim as optim
from tqdm import tqdm
from sklearn import metrics


class SentimentDetectionDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = [os.path.join(data_path, data) for data in os.listdir(data_path)]

    def __getitem__(self, item):
        file_path = self.file_list[item]
        data = torch.load(file_path)
        id_index = data["id"]
        text_encoder = data["text_encoder"]
        chatglm_generate_text_encoder = data["chatglm_generate_text_encoder"]
        image_encoder = data["image_encoder"]
        cogvlm_generate_text_encoder = data["cogvlm_generate_text_encoder"]
        label = data["label"]
        return id_index, text_encoder, chatglm_generate_text_encoder, image_encoder, cogvlm_generate_text_encoder, label

    def __len__(self):
        return len(self.file_list)


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, head_num):
        super().__init__()
        self.linear_input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_output = nn.Linear(hidden_dim,output_dim)

    def forward(self, feature):

        feature = self.linear_input(feature)
        feature = self.activate(feature)
        feature = self.layer_norm_input(feature)
        feature = self.dropout(feature)
        feature = self.linear_output(feature)
        return feature


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear_input = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        feature = self.linear_input(feature)
        feature = self.activate(feature)
        feature = self.layer_norm_input(feature)
        feature = self.dropout(feature)
        feature = self.linear_output(feature)
        return feature


class Map_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear_input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.activate = nn.GELU()
        self.dropout_input = nn.Dropout(p=dropout)
        self.linear_output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, feature):
        feature = self.linear_input(feature)
        feature = self.activate(feature)
        feature = self.layer_norm_input(feature)
        feature = self.dropout_input(feature)
        feature = self.linear_output(feature)
        return feature

class Encoder(nn.Module):
    def __init__(self,feature_dim,head,dropout):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = head
        self.dropout = dropout
        self.multi_head_attention = nn.MultiheadAttention(feature_dim,head,dropout=self.dropout,batch_first=True)
        self.attn_dropout = nn.Dropout(p=self.dropout)
        self.layer_norm_input = nn.LayerNorm(self.feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.feature_dim,self.feature_dim*2),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_dim*2,self.feature_dim)
        )
        self.layer_norm_output = nn.LayerNorm(self.feature_dim)
        self.fn_dropout = nn.Dropout(p=self.dropout)

    def forward(self,q,k,v):
        residual = q
        attn_output,_ = self.multi_head_attention(q,k,v)
        attn_output = self.attn_dropout(attn_output)
        attn_output = attn_output+residual
        residual = attn_output
        x = self.feed_forward(attn_output)
        x = self.fn_dropout(x)
        x = x+residual
        x = self.layer_norm_output(x)
        return x


class SentimentDetectionClassifer(nn.Module):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser.parse_args()
        self.text_adapter = Adapter(input_dim=self.parser.text_input_dim, hidden_dim=self.parser.text_hidden_dim,
                                    output_dim=self.parser.text_output_dim, dropout=self.parser.dropout,
                                    head_num=self.parser.head_num)
        self.chatglm_generate_text_adapter = Adapter(input_dim=self.parser.chatglm_generate_text_input_dim,
                                                     hidden_dim=self.parser.chatglm_generate_text_hidden_dim,
                                                     output_dim=self.parser.chatglm_generate_text_output_dim,
                                                     dropout=self.parser.dropout, head_num=self.parser.head_num)
        self.image_adapter = Adapter(input_dim=self.parser.image_input_dim, hidden_dim=self.parser.image_hidden_dim,
                                     output_dim=self.parser.image_output_dim, dropout=self.parser.dropout,
                                     head_num=self.parser.head_num)
        self.cogvlm_generate_text_adapter = Adapter(input_dim=self.parser.cogvlm_generate_text_input_dim,
                                                    hidden_dim=self.parser.cogvlm_generate_text_hidden_dim,
                                                    output_dim=self.parser.cogvlm_generate_text_output_dim,
                                                    dropout=self.parser.dropout, head_num=self.parser.head_num)
        self.text_multi_head_attention = nn.ModuleList(
            [Encoder(self.parser.text_output_dim,self.parser.head_num,self.parser.dropout) for _ in range(self.parser.layer_num)]
        )

        self.image_text_multi_head_attention = nn.ModuleList(
            [Encoder(self.parser.text_output_dim,self.parser.head_num,self.parser.dropout) for _ in range(self.parser.layer_num)]
        )

        self.image_cogvlm_multi_head_attention = nn.ModuleList(
            [Encoder(self.parser.text_output_dim,self.parser.head_num,self.parser.dropout) for _ in range(self.parser.layer_num)]
        )

        self.cross_text_multi_head_attention = nn.ModuleList(
            [Encoder(self.parser.text_output_dim,self.parser.head_num,self.parser.dropout) for _ in range(self.parser.layer_num)]
        )

        self.cross_image_multi_head_attention = nn.ModuleList(
            [Encoder(self.parser.text_output_dim,self.parser.head_num,self.parser.dropout) for _ in range(self.parser.layer_num)]
        )
        self.mlp = MLP(input_dim=self.parser.text_output_dim, hidden_dim=256, output_dim=self.parser.class_num, dropout=self.parser.dropout)
        self.w = nn.Parameter(torch.empty((self.parser.image_output_dim, self.parser.text_output_dim)))
        nn.init.kaiming_normal_(self.w,nonlinearity="relu")
        self.full_network = Map_Network(self.parser.text_output_dim+self.parser.image_output_dim,512,self.parser.text_output_dim,self.parser.dropout)

    def forward(self, id_value, text_encoder, chatglm_generate_text_encoder, image_encoder,
                cogvlm_generate_text_encoder):
        text_encoder = text_encoder.to(torch.float)
        chatglm_generate_text_encoder = chatglm_generate_text_encoder.to(torch.float)
        image_encoder = image_encoder.to(torch.float)
        cogvlm_generate_text_encoder = cogvlm_generate_text_encoder.to(torch.float)
        text_encoder = text_encoder.reshape((text_encoder.shape[0], text_encoder.shape[2], text_encoder.shape[3]))
        chatglm_generate_text_encoder = chatglm_generate_text_encoder.reshape((chatglm_generate_text_encoder.shape[0],
                                                                               chatglm_generate_text_encoder.shape[2],
                                                                               chatglm_generate_text_encoder.shape[3]))
        image_encoder = image_encoder.reshape((image_encoder.shape[0], image_encoder.shape[2], image_encoder.shape[3]))
        cogvlm_generate_text_encoder = cogvlm_generate_text_encoder.reshape((cogvlm_generate_text_encoder.shape[0],
                                                                             cogvlm_generate_text_encoder.shape[2],
                                                                             cogvlm_generate_text_encoder.shape[3]))

        text_encoder = self.text_adapter(text_encoder)
        chatglm_generate_text_encoder = self.chatglm_generate_text_adapter(chatglm_generate_text_encoder)

        image_encoder = self.image_adapter(image_encoder)
        cogvlm_generate_text_encoder = self.cogvlm_generate_text_adapter(cogvlm_generate_text_encoder)

        text_chatglm_encoder = text_encoder
        for i in range(self.parser.layer_num):
            text_chatglm_encoder = self.text_multi_head_attention[i](text_chatglm_encoder,chatglm_generate_text_encoder,chatglm_generate_text_encoder)

        x = torch.matmul(image_encoder, self.w)
        text_encoder_transpose = text_encoder.transpose(1, 2)
        y = torch.matmul(x, text_encoder_transpose)

        text_encoder_cls = text_encoder[:, 0, :]
        text_encoder_pad = text_encoder_cls.unsqueeze(1)
        text_encoder_pad = text_encoder_pad.repeat(1, image_encoder.shape[1], 1)
        image_encoder_text = torch.concat((image_encoder, text_encoder_pad), dim=-1)
        image_encoder_text = self.full_network(image_encoder_text)
        matmul_text = y[:, :, 0]
        matmul_text = matmul_text.reshape((matmul_text.shape[0], matmul_text.shape[1], 1))
        image_encoder_text = image_encoder_text + matmul_text

        image_encoder_cls = image_encoder[:, 0, :]
        image_encoder_pad = image_encoder_cls.unsqueeze(1)
        image_encoder_pad = image_encoder_pad.repeat(1, text_encoder.shape[1], 1)
        text_encoder_image = torch.concat((text_encoder, image_encoder_pad), dim=-1)
        text_encoder_image = self.full_network(text_encoder_image)
        matmul_image = y[:, 0, :]
        matmul_image = matmul_image.reshape((matmul_image.shape[0], matmul_image.shape[1], 1))
        text_encoder_image = text_encoder_image + matmul_image

        image_text_encoder = image_encoder_text
        image_cogvlm_encoder = text_encoder_image

        for i in range(self.parser.layer_num):
            image_text_encoder = self.image_text_multi_head_attention[i](image_text_encoder,text_chatglm_encoder,text_chatglm_encoder)

        for i in range(self.parser.layer_num):
            image_cogvlm_encoder = self.image_cogvlm_multi_head_attention[i](image_cogvlm_encoder,cogvlm_generate_text_encoder,cogvlm_generate_text_encoder)


        cross_text_encoder = image_text_encoder
        for i in range(self.parser.layer_num):
            cross_text_encoder = self.cross_text_multi_head_attention[i](cross_text_encoder,image_cogvlm_encoder,image_cogvlm_encoder)

        cross_image_encoder = image_cogvlm_encoder
        for i in range(self.parser.layer_num):
            cross_image_encoder = self.cross_image_multi_head_attention[i](cross_image_encoder,image_text_encoder,image_text_encoder)


        pad_image_encoder = torch.zeros(
            (cross_text_encoder.shape[0], cross_text_encoder.shape[1], cross_text_encoder.shape[2]))
        pad_image_encoder = pad_image_encoder.to(self.parser.device)
        pad_image_encoder[:, 0:cross_image_encoder.shape[1], :] = cross_image_encoder
        cross_image_encoder = pad_image_encoder

        total_encoder = cross_text_encoder + cross_image_encoder

        inputs = torch.mean(total_encoder,dim=1)
        outputs = self.mlp(inputs)
        chatglm_inputs = torch.mean(image_text_encoder,dim=1)
        chatglm_output = self.mlp(chatglm_inputs)
        cross_image_inputs = torch.mean(image_cogvlm_encoder,dim=1)
        cross_image_output = self.mlp(cross_image_inputs)
        return outputs,chatglm_output,cross_image_output


def load_data(data_path):
    data = []
    for file_name in tqdm(os.listdir(data_path), total=len(os.listdir(data_path)), desc="加载数据"):
        file_path = os.path.join(data_path, file_name)
        data_encoder = torch.load(file_path)
        data.append(data_encoder)
    return data


def train(parser):
    arg_parser = parser.parse_args()
    lr = arg_parser.lr
    epochs = arg_parser.epochs
    early_stop = arg_parser.early_stop
    device = arg_parser.device
    model = SentimentDetectionClassifer(parser).to(device)
    criterion = nn.CrossEntropyLoss()
    batch_size = arg_parser.batch_size
    random_seed = arg_parser.random_seed
    torch.manual_seed(random_seed)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=0.07)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    data_path = arg_parser.train_data_path
    train_dataset = SentimentDetectionDataSet(data_path)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_path = arg_parser.test_data_path
    valid_data_path = arg_parser.valid_data_path
    model_save_path = arg_parser.model_save_path
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(epochs):
        model = model.train()
        total_correct = 0
        total_sample = 0
        predicted_labels = []
        truth_labels = []
        total_loss = 0
        total_batch = 0
        for batch_index, (
                id_index, text_encoder, chatglm_generate_text_encoder, image_encoder, cogvlm_generate_text_encoder,
                label) in enumerate(train_data_loader):
            id_index = id_index.to(device)
            text_encoder = text_encoder.to(device)
            chatglm_generate_text_encoder = chatglm_generate_text_encoder.to(device)
            image_encoder = image_encoder.to(device)
            cogvlm_generate_text_encoder = cogvlm_generate_text_encoder.to(device)
            label = label.to(device)
            # print("开始训练")
            outputs,chatglm_outputs,cross_image_outputs = model(id_index, text_encoder, chatglm_generate_text_encoder, image_encoder,
                            cogvlm_generate_text_encoder)
            # print("训练完成")
            _, predict_labels = torch.max(outputs, 1)
            correct = (predict_labels == label).sum().item()
            total_correct += correct
            total_sample += len(label)
            predicted_labels.extend(predict_labels.tolist())
            truth_labels.extend(label.tolist())

            loss = criterion(outputs, label)
            output_loss = loss
            chatglm_loss = criterion(chatglm_outputs,label)
            cross_image_loss = criterion(cross_image_outputs,label)
            loss = loss+0.3*chatglm_loss+0.3*cross_image_loss
            # loss = loss+l2_loss

            # print("反向传播")
            loss.backward()
            # print("反向完成")
            optimizer.step()
            optimizer.zero_grad()
            if batch_index % 100 == 0:
                print(
                    f'Train Epoch:[{epoch + 1}/{epochs}],Train Step:[{batch_index + 1}/{len(train_data_loader)}],Train Loss:[{loss.item()}]')
                print(f"loss:{output_loss},chatglm_loss:{chatglm_loss},cross_image_loss:{cross_image_loss}")
            total_loss += loss.item()
            total_batch += 1
        total_loss = total_loss / total_batch
        #scheduler.step(total_loss)
        print(
            f'Train Epoch:[{epoch + 1}/{epochs}],total_correct:{total_correct},total_sample:{total_sample},accuracy:{total_correct / total_sample}')
        #epoch_accuracy = test(valid_data_path, model, batch_size, device)
        epoch_accuracy,epoch_loss = evaluate(valid_data_path,model,batch_size,device,criterion)
        scheduler.step(epoch_loss)
        if epoch_accuracy > best_accuracy:
            torch.save(model, model_save_path)
            best_epoch = 0
            best_accuracy = epoch_accuracy
        else:
            best_epoch += 1
        if best_epoch >= early_stop:
            break
    model = torch.load(model_save_path)
    print("test_data:")
    test(test_data_path, model, batch_size, device)

def evaluate(data_path,model,batch_size,device,criterion):
    model = model.to(device).eval()
    valid_dataset = SentimentDetectionDataSet(data_path)
    valid_data_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
    total_loss = 0
    correct = 0
    predicted_labels = []
    truth_labels = []
    with torch.no_grad():
        for id_index,text_encoder,chatglm_generate_text_encoder,image_encoder,cogvlm_generate_text_encoder,label in valid_data_loader:
            id_index = id_index.to(device)
            text_encoder = text_encoder.to(device)
            chatglm_generate_text_encoder = chatglm_generate_text_encoder.to(device)
            image_encoder = image_encoder.to(device)
            cogvlm_generate_text_encoder = cogvlm_generate_text_encoder.to(device)
            label = label.to(device)
            outputs,chatglm_outputs,cross_image_outputs = model(id_index,text_encoder,chatglm_generate_text_encoder,image_encoder,cogvlm_generate_text_encoder)
            loss = criterion(outputs,label)
            chatglm_loss =criterion(chatglm_outputs,label)
            cross_image_loss = criterion(cross_image_outputs,label)
            loss = loss+0.3*chatglm_loss+0.3*cross_image_loss
            total_loss+=loss.item()
            _,predicted_label = torch.max(outputs,dim=1)
            truth_labels.extend(label.tolist())
            predicted_labels.extend(predicted_label.tolist())
            correct+=(label==predicted_label).sum()
    accuracy = correct/len(valid_data_loader.dataset)
    macro_f1 = metrics.f1_score(truth_labels,predicted_labels,average="macro")
    macro_precision = metrics.precision_score(truth_labels,predicted_labels,average="macro")
    macro_recall = metrics.recall_score(truth_labels,predicted_labels,average="macro")
    micro_f1 = metrics.f1_score(truth_labels,predicted_labels,average="micro")
    micro_precision = metrics.precision_score(truth_labels,predicted_labels,average="micro")
    micro_recall = metrics.recall_score(truth_labels,predicted_labels,average="micro")
    binary_f1 = metrics.f1_score(truth_labels,predicted_labels,average="binary")
    binary_precision = metrics.precision_score(truth_labels,predicted_labels,average="binary")
    binary_recall = metrics.recall_score(truth_labels,predicted_labels,average="binary")
    print(f"correct_sample:{correct},total_sample:{len(valid_data_loader.dataset)},accuracy:{accuracy},macro_f1:{macro_f1},macro_precision:{macro_precision},macro_recall:{macro_recall},micro_f1:{micro_f1},micro_precision:{micro_precision},micro_recall:{micro_recall},binary_f1:{binary_f1},binary_precision:{binary_precision},binary_recall:{binary_recall}")
    print(f"total_loss:{total_loss}")
    confusion_matrix = metrics.confusion_matrix(truth_labels,predicted_labels)
    print(confusion_matrix)
    return accuracy,total_loss
def test(data_path, model, batch_size, device):
    model = model.to(device).eval()
    correct = 0
    # test_data = load_data(data_path)
    test_dataset = SentimentDetectionDataSet(data_path)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predicted_labels = []
    truth_labels = []
    correct_label_id = []
    error_label_id = []
    with torch.no_grad():
        for id_value, text_encoder, chatglm_generate_text_encoder, image_encoder, cogvlm_generate_text_encoder, label in test_data_loader:
            id_value = id_value.to()
            text_encoder = text_encoder.to(device)
            chatglm_generate_text_encoder = chatglm_generate_text_encoder.to(device)
            image_encoder = image_encoder.to(device)
            cogvlm_generate_text_encoder = cogvlm_generate_text_encoder.to(device)
            label = label.to(device)
            outputs,chatglm_outputs,cross_image_outputs = model(id_value, text_encoder, chatglm_generate_text_encoder, image_encoder,
                            cogvlm_generate_text_encoder)
            _, predict_label = torch.max(outputs, 1)
            truth_labels.extend(label.tolist())
            predicted_labels.extend(predict_label.tolist())
            correct += (label == predict_label).sum()

            id_list = id_value.tolist()
            truth_label_list = label.tolist()
            predict_label_list = predict_label.tolist()
            for i in range(len(id_list)):
                if truth_label_list[i] == predict_label_list[i]:
                    correct_label_id.append(id_list[i])
                else:
                    error_label_id.append(id_list[i])

    accuracy = correct / len(test_data_loader.dataset)
    macro_f1 = metrics.f1_score(truth_labels, predicted_labels, average="macro")
    macro_precision = metrics.precision_score(truth_labels, predicted_labels, average="macro")
    macro_recall = metrics.recall_score(truth_labels, predicted_labels, average="macro")
    micro_f1 = metrics.f1_score(truth_labels,predicted_labels,average="micro")
    micro_precision = metrics.precision_score(truth_labels,predicted_labels,average="micro")
    micro_recall = metrics.recall_score(truth_labels,predicted_labels,average="micro")
    binary_f1 = metrics.f1_score(truth_labels,predicted_labels,average="binary")
    binary_precision = metrics.precision_score(truth_labels,predicted_labels,average="binary")
    binary_recall = metrics.recall_score(truth_labels,predicted_labels,average="binary")

    print(
        f'total_correct:{correct},total_sample:{len(test_data_loader.dataset)},accuracy:{accuracy},macro_f1:{macro_f1},macro_precision:{macro_precision},macro_recall:{macro_recall},micro_f1:{micro_f1},micro_precision:{micro_precision},micro_recall:{micro_recall},binary_f1:{binary_f1},binary_precision:{binary_precision},binary_recall:{binary_recall}')
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SentimentDetection")
    parser.add_argument('--text_input_dim', default=4096)
    parser.add_argument('--text_hidden_dim', default=2048)
    parser.add_argument('--text_output_dim', default=1024)
    parser.add_argument('--chatglm_generate_text_input_dim', default=4096)
    parser.add_argument('--chatglm_generate_text_hidden_dim', default=2048)
    parser.add_argument('--chatglm_generate_text_output_dim', default=1024)
    parser.add_argument('--image_input_dim', default=768)
    parser.add_argument('--image_hidden_dim', default=768)
    parser.add_argument('--image_output_dim', default=768)
    parser.add_argument('--text_max_length', default=100)
    parser.add_argument('--cogvlm_generate_text_input_dim', default=4096)
    parser.add_argument('--cogvlm_generate_text_hidden_dim', default=2048)
    parser.add_argument('--cogvlm_generate_text_output_dim', default=1024)
    parser.add_argument('--class_num', default=2)
    # parser.add_argument('--lr', default=3e-5)
    parser.add_argument('--lr', default=0.00003)
    parser.add_argument('--epochs', default=300)
    parser.add_argument('--early_stop', default=10)
    parser.add_argument('--batch_size', default=40)
    parser.add_argument('--l2_lambda', default=0.01)
    parser.add_argument('--head_num', default=8)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--layer_num', default=3)
    parser.add_argument('--train_data_path', default='./result/encoder/HFM/total_encoder/train/')
    parser.add_argument('--test_data_path', default='./result/encoder/HFM/total_encoder/test/')
    parser.add_argument('--valid_data_path', default='./result/encoder/HFM/total_encoder/valid/')
    parser.add_argument('--device', default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model_save_path', default='./model/HFM/model_test_2024_11_23_15_09_C.pth')
    parser.add_argument('--random_seed', default=100)
    train(parser)
