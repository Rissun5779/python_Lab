from keras.utils import text_dataset_from_directory

# 建立訓練,驗證和測試資料集
batch_size = 32
raw_train_ds, raw_val_ds = text_dataset_from_directory(
    "/Dataset/aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="both",
    seed=1337
)
raw_test_ds = text_dataset_from_directory(
    "/Dataset/aclImdb/test", 
    batch_size=batch_size,
    seed=1337
)
print("--------------------------")
# 顯示3個資料集的第1筆資料
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(1):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
print("--------------------------")
for text_batch, label_batch in raw_val_ds.take(1):
    for i in range(1):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
print("--------------------------")
for text_batch, label_batch in raw_test_ds.take(1):
    for i in range(1):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])