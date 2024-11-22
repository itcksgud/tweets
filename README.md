Natural Language Processing with Disaster Tweets
=============
Predict which Tweets are about real disasters and which ones are not
-------------
# 코드 수행 과정

### 필수 라이브러리 임포트
```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
import re
```

### 데이터 전처리
```python
# csv파일에서 df_train 저장
df_train = pd.read_csv(r'C:\Users\kingc\Desktop\ai\nlp-getting-started\train.csv')
df_real=pd.read_csv(r'C:\Users\kingc\Desktop\ai\nlp-getting-started\test.csv')
df_real['target']=0

# 데이터 전처리
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # URL 제거
    text = re.sub(r'@\w+', '', text)    # 멘션 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # 특수문자 제거
    text = text.lower().strip()        # 소문자 변환 및 양쪽 공백 제거
    return text

df_train['text'] = df_train['text'].apply(clean_text)
df_real['text'] = df_real['text'].apply(clean_text)
```

### train, test 분리
```python
# df_train->train, test로 분리
train_input, test_input, train_target, test_target = train_test_split(
    df_train['text'], df_train['target'], test_size=0.2, random_state=42)

# 인덱스를 리셋하여 연속적인 숫자로 설정
train_input.reset_index(drop=True, inplace=True)
train_target.reset_index(drop=True, inplace=True)
```


### 데이터셋 클래스 정의
```python
# 데이터셋 클래스 정의
class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        target = self.targets.iloc[idx]


        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.long)
        }
```


### Bert 모델 클래스 정의
```python
class DistilbertBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased'):
        super(DistilbertBinaryClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.5)  # Dropout 비율을 증가하여 과적합 방지
        self.fc = nn.Linear(self.distilbert.config.hidden_size, 1) 

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS 
        dropout_output = self.dropout(pooled_output)
        return self.fc(dropout_output)  # 로짓 출력
```

### 하이퍼 파라미터 설정 및 데이터 로더 생성
```python
# 하이퍼파라미터 설정
PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

# distilbert 토크나이저 초기화
tokenizer =  DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 데이터셋 준비
train_dataset = TextDataset(train_input, train_target, tokenizer, MAX_LEN)
test_dataset = TextDataset(test_input, test_target, tokenizer, MAX_LEN)
real_dataset = TextDataset(df_real['text'], df_real['target'], tokenizer, MAX_LEN)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilbertBinaryClassifier(pretrained_model_name=PRETRAINED_MODEL_NAME)
model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.BCEWithLogitsLoss()  # 이진 분류 손실 함수
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

### 학습 및 평가
```python
best_f1 = 0  # 최고의 F1 Score를 저장할 변수
best_accuracy = 0  # 최고의 Accuracy를 저장할 변수
best_epoch = 0  # 최고의 F1 Score를 기록한 에포크 번호

for epoch in range(EPOCHS):
    # ====== 학습 단계 ======
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device).float()  # BCEWithLogitsLoss는 float 입력 필요

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(-1), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # ====== 평가 단계 ======
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            # 예측값 계산
            outputs = model(input_ids, attention_mask)
            predictions = torch.sigmoid(outputs).squeeze(-1) > 0.5  # 0.5 기준으로 이진 분류

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # F1 Score 및 정확도 계산
    f1 = f1_score(all_targets, all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Epoch {epoch + 1}, Test F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    

    # F1 Score 비교 및 저장
    if f1 > best_f1:
        best_f1 = f1
        best_accuracy = accuracy
        best_epoch = epoch + 1
        # 모델 저장 (옵션)
        torch.save(model.state_dict(), "best_model.pth")
        print(f"New best F1 Score: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}, Model saved.")

# 학습 종료 후 최고의 F1 Score와 에포크 출력
print(f"Best F1 Score: {best_f1:.4f} at Epoch {best_epoch}")
print(f"Best Accuracy: {best_accuracy:.4f} at Epoch {best_epoch}")
```

## 모델 예측 수행 및 결과물 생성
```python
# 모델 예측 수행
model.eval()
all_preds = []

with torch.no_grad():  # No gradient computation for prediction
    for batch in real_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 모델 예측
        outputs = model(input_ids, attention_mask)

        # 예측을 0 또는 1로 변환 (이진 분류)
        preds = torch.round(torch.sigmoid(outputs)).squeeze(-1).cpu().numpy()
        all_preds.extend(preds)

all_preds=list(map(int,all_preds))
sample_submission = pd.read_csv(r'C:\Users\kingc\Desktop\ai\nlp-getting-started\sample_submission.csv')
sample_submission["target"] = all_preds
sample_submission.to_csv("submission.csv", index=False)

```


# 문제점
## EPOCH를 돌아도 f1-score가 좋아지지 않음
aug: nltk를 이용해 train data를 늘렸을 때

aug=True, Dropout=0.3, Learning_Late=2e-5, EPOCHS=5, Bert

Epoch 1, Loss: 0.38173519467978967
Epoch 1, Test F1 Score: 0.7964, Accuracy: 0.8345
New best F1 Score: 0.7964, Accuracy: 0.8345, Model saved.
Epoch 2, Loss: 0.19602288163494408
Epoch 2, Test F1 Score: 0.7796, Accuracy: 0.7984
Epoch 3, Loss: 0.08535619048717043
Epoch 3, Test F1 Score: 0.7935, Accuracy: 0.8247
Epoch 4, Loss: 0.062152522151190985
Epoch 4, Test F1 Score: 0.7804, Accuracy: 0.8201
Epoch 5, Loss: 0.04369633622040356
Epoch 5, Test F1 Score: 0.7674, Accuracy: 0.8221

Best F1 Score: 0.7964 at Epoch 1
Best Accuracy: 0.8345 at Epoch 1


aug=True, Dropout=0.5, Learning_Late=1e-5, EPOCHS=5, distilbert

Epoch 1, Loss: 0.4088034744807116
Epoch 1, Test F1 Score: 0.7814, Accuracy: 0.7991
New best F1 Score: 0.7814, Accuracy: 0.7991, Model saved.
Epoch 2, Loss: 0.2807245238322344
Epoch 2, Test F1 Score: 0.7851, Accuracy: 0.8260
New best F1 Score: 0.7851, Accuracy: 0.8260, Model saved.
Epoch 3, Loss: 0.1707592908258787
Epoch 3, Test F1 Score: 0.7820, Accuracy: 0.8155
Epoch 4, Loss: 0.10306167098622662
Epoch 4, Test F1 Score: 0.7827, Accuracy: 0.8155
Epoch 5, Loss: 0.06627417067243055
Epoch 5, Test F1 Score: 0.7691, Accuracy: 0.8056

Best F1 Score: 0.7851 at Epoch 2
Best Accuracy: 0.8260 at Epoch 2

aug=True, Dropout=0.3, Learning_Late=2e-5, EPOCHS=5, distilbert

Epoch 1, Loss: 0.3947889788511388
Epoch 1, Test F1 Score: 0.8089, Accuracy: 0.8418
New best F1 Score: 0.8089, Accuracy: 0.8418, Model saved.
Epoch 2, Loss: 0.21290159154331356
Epoch 2, Test F1 Score: 0.7926, Accuracy: 0.8168
Epoch 3, Loss: 0.09948483921998129
Epoch 3, Test F1 Score: 0.7708, Accuracy: 0.8043
Epoch 4, Loss: 0.06156847309831437
Epoch 4, Test F1 Score: 0.7736, Accuracy: 0.8083
Epoch 5, Loss: 0.04882351213369795
Epoch 5, Test F1 Score: 0.7812, Accuracy: 0.8168

Best F1 Score: 0.8089 at Epoch 1
Best Accuracy: 0.8418 at Epoch 1


aug=True, Dropout=0.5, Learning_Late=2e-5, EPOCHS=5, distilbert

Epoch 1, Loss: 0.03891518039825486
Epoch 1, Test F1 Score: 0.7740, Accuracy: 0.8148
New best F1 Score: 0.7740, Accuracy: 0.8148, Model saved.
Epoch 2, Loss: 0.033381932591715416
Epoch 2, Test F1 Score: 0.7835, Accuracy: 0.8207
New best F1 Score: 0.7835, Accuracy: 0.8207, Model saved.
Epoch 3, Loss: 0.032929053397563815
Epoch 3, Test F1 Score: 0.7731, Accuracy: 0.8142
Epoch 4, Loss: 0.03131099633023151
Epoch 4, Test F1 Score: 0.7806, Accuracy: 0.8162
Epoch 5, Loss: 0.0317130947469601
Epoch 5, Test F1 Score: 0.7709, Accuracy: 0.8056

Best F1 Score: 0.7835 at Epoch 2
Best Accuracy: 0.8207 at Epoch 2

aug=True, Dropout=0.1, Learning_Late=2e-5, EPOCHS=5, distilbert

Epoch 1, Loss: 0.38240285943264724
Epoch 1, Test F1 Score: 0.7969, Accuracy: 0.8437
New best F1 Score: 0.7969, Accuracy: 0.8437, Model saved.
Epoch 2, Loss: 0.20561490254598888
Epoch 2, Test F1 Score: 0.7912, Accuracy: 0.8247
Epoch 3, Loss: 0.09057112464900043
Epoch 3, Test F1 Score: 0.7668, Accuracy: 0.8043
Epoch 4, Loss: 0.058654142358508986
Epoch 4, Test F1 Score: 0.7657, Accuracy: 0.7899
Epoch 5, Loss: 0.045385472161076315
Epoch 5, Test F1 Score: 0.7848, Accuracy: 0.8135

Best F1 Score: 0.7969 at Epoch 1
Best Accuracy: 0.8437 at Epoch 1

## BERT Large를 사용해도 성능차이가 확연하기 드러나지 않음.
aug=False, Dropout=0.5, Learning_Late=2e-5, EPOCHS=2, Bert Large

Epoch 1, Loss: 0.42919120742032535
TP: 501, TN: 766, FP: 108, FN: 148
Epoch 1, Test F1 Score: 0.7965, Accuracy: 0.8319
New best F1 Score: 0.7965, Accuracy: 0.8319, Model saved.
Epoch 2, Loss: 0.30344746760496005
TP: 503, TN: 774, FP: 100, FN: 146
Epoch 2, Test F1 Score: 0.8035, Accuracy: 0.8385
New best F1 Score: 0.8035, Accuracy: 0.8385, Model saved.

Best F1 Score: 0.8035 at Epoch 2
Best Accuracy: 0.8385 at Epoch 2

실행 시간: 114m 10.4s

# 제출 결과
![image](https://github.com/user-attachments/assets/776061b9-7a1a-41a9-be81-e5dd75373908)

0.83726으로 1000명이 넘는 참가자 중 71등에 올라감.

