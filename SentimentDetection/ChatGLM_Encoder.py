import torch
from transformers import AutoTokenizer,AutoModel
import pandas as pd
from tqdm import tqdm

class ChatGLM_Encoder_Class:
    def __init__(self,pretrained_model_path,device):
        self.pretrained_model_path = pretrained_model_path
        self.device = device
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path,trust_remote_code=True)
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_path,trust_remote_code=True).to(self.device)
    def encoder(self,dataset_path,save_path,index_value,column_value,max_length):
        data_list = []
        data = pd.read_excel(dataset_path)
        for index,row in data.iterrows():
            #if index != 0:
            id = row[index_value]
            text = row[column_value]
            '''image = row["image"]
            label = row["label"]'''
            data_list.append([id,text])
        for row in tqdm(data_list,total=len(data_list),desc="处理进度"):
            text_index = row[0]
            text_value = row[1]
            inputs = self.pretrained_tokenizer(str(text_value),return_tensors="pt",padding="max_length",max_length=max_length,truncation=True)
            inputs = inputs.to(self.device)
            outputs = self.pretrained_model.transformer(**inputs)
            outputs = outputs.last_hidden_state
            text_encoder = outputs
            generate_encoder = {"text_index":text_index,"text_value":text_value,"text_encoder":text_encoder}
            torch.save(generate_encoder,save_path+str(text_index)+".pkl")




if __name__ == '__main__':
    pretrained_model_path = "../../pretrained_model/glm-4-9b-chat"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    chatglm_encoder_class = ChatGLM_Encoder_Class(pretrained_model_path,device)
    max_length = 100
    index_value = "id"
    column_value = "text"
    train_dataset_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/train.xlsx"
    train_save_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/train/"
    chatglm_encoder_class.encoder(train_dataset_path,train_save_path,index_value,column_value,max_length)
    test_dataset_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/test.xlsx"
    test_save_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/test/"
    chatglm_encoder_class.encoder(test_dataset_path,test_save_path,index_value,column_value,max_length)
    valid_dataset_path = "./DataSet/SarcNet/SarcNet-Image-Text/text/valid.xlsx"
    valid_save_path = "./result/SarcNet/SarcNet-Image-Text/text_encoder/valid/"
    chatglm_encoder_class.encoder(valid_dataset_path,valid_save_path,index_value,column_value,max_length)

    index_value_chatglm_generate = "id"
    column_value_chatglm_generate = "chatglm_generate"
    max_length_chatglm_generate = 100
    train_dataset_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate/train.xlsx"
    train_save_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/train/"
    chatglm_encoder_class.encoder(train_dataset_chatglm_generate_path,train_save_chatglm_generate_path,index_value_chatglm_generate,column_value_chatglm_generate,max_length_chatglm_generate)
    test_dataset_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate/test.xlsx"
    test_save_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/test/"
    chatglm_encoder_class.encoder(test_dataset_chatglm_generate_path,test_save_chatglm_generate_path,index_value_chatglm_generate,column_value_chatglm_generate,max_length_chatglm_generate)
    valid_dataset_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate/valid.xlsx"
    valid_save_chatglm_generate_path = "./result/SarcNet/SarcNet-Image-Text/chatglm_generate_encoder/valid/"
    chatglm_encoder_class.encoder(valid_dataset_chatglm_generate_path,valid_save_chatglm_generate_path,index_value_chatglm_generate,column_value_chatglm_generate,max_length_chatglm_generate)

    index_value_cogvlm_generate = "id"
    column_value_cogvlm_generate = "image_generate"
    max_length_cogvlm_generate = 100
    train_dataset_cogvlm_generate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate/train.xlsx"
    train_save_cogvlm_generate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/train/"
    chatglm_encoder_class.encoder(train_dataset_cogvlm_generate_path,train_save_cogvlm_generate_path,index_value_cogvlm_generate,column_value_cogvlm_generate,max_length_cogvlm_generate)
    test_dataset_cogvlm_generate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate/test.xlsx"
    test_save_cogvlm_generate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/test/"
    chatglm_encoder_class.encoder(test_dataset_cogvlm_generate_path,test_save_cogvlm_generate_path,index_value_cogvlm_generate,column_value_cogvlm_generate,max_length_cogvlm_generate)
    valid_dataset_cogvlm_geberate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate/valid.xlsx"
    valid_save_cogvlm_generate_path = "./result/SarcNet/SarcNet-Image-Text/cogvlm_generate_encoder/valid/"
    chatglm_encoder_class.encoder(valid_dataset_cogvlm_geberate_path,valid_save_cogvlm_generate_path,index_value_cogvlm_generate,column_value_cogvlm_generate,max_length_cogvlm_generate)










