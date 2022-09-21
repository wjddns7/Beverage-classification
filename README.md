# 음료 분류 AI

## 1. 제작 기간
* 2022.07.14 ~ 2022.07.15
<br/>

## 2. 제작 도구
* Jupyter
<br/>

## 3. 핵심 기능
* 미리 학습 시켜둔 음료사진을 이용하여 인터넷에 있는 사진과 비교하여 그 음료가 어떤 음료인지 예측합니다.
<br/>

## 4. 주요 코드
### 데이터를 불러오고 이미지 크기 변환
```
import pathlib
data_dir = r'C:\Users\user\Downloads\drink\drink'  # 폴더 불러오기
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.png')))
print(image_count)

#이미지 크기 변환
batch_size = 32
img_height = 200
img_width = 200
```
### 데이터를 train set와 test set으로 나누기
```
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=256,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  ```
### 이미지 확인
  ```
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  ```
  ![image](https://user-images.githubusercontent.com/108790183/191388779-a6d620f7-d970-4a42-9497-f6bb58dba3b6.png)
### 이미지를 정규화
```
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
```
### Sequential 모델 레이어 구축
```
#sequential모델 레이어 구축
num_classes = 16

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```
### 모델 컴파일
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
### 레이어 확인
![image](https://user-images.githubusercontent.com/108790183/191389843-58c301dd-f6c0-4424-a8f3-82e350caa320.png)
### 모델 훈련
```
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```
### 정확도 및 손실 확인
```
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
![image](https://user-images.githubusercontent.com/108790183/191390135-a64650cf-892f-4749-982f-0046941d6d17.png)
### 예측
```
test_dir = r"C:\Users\user\Downloads\drink\drink\6\201.jpg"
test_dir = pathlib.Path(test_dir)

img = keras.preprocessing.image.load_img(test_dir, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
```
![image](https://user-images.githubusercontent.com/108790183/191390250-1b6d633c-1829-4f9f-9407-5ad5b308dba3.png)
## 5. 어려웠던 점
#### 음료 데이터셋을 찾기가 생각보다 힘들어서 Teachable Machine을 이용해서 직접 데이터셋을 만들었습니다.
