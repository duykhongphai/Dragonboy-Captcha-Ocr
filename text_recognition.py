import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# ========================= CHUẨN BỊ DỮ LIỆU =========================

def load_captcha_images(folder_path):
    """
    Đọc hình ảnh CAPTCHA và nhãn từ tên file
    
    Args:
        folder_path: Đường dẫn đến thư mục chứa hình ảnh CAPTCHA
        
    Returns:
        images: Danh sách các hình ảnh đã xử lý
        labels: Danh sách các nhãn tương ứng
    """
    images = []
    labels = []
    
    # Lấy tất cả các file trong thư mục
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for file_name in files:
        # Lấy nhãn từ tên file
        label = file_name.split('.')[0]  # Lấy phần trước dấu chấm
        
        # Đọc hình ảnh
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Xử lý hình ảnh
            img = preprocess_image(img)
            
            # Thêm vào danh sách
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_image(image):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Chuẩn hóa kích thước
    resized = cv2.resize(gray, (128, 64))
    
    # Chuẩn hóa pixel values
    normalized = resized / 255.0
    
    # Thêm kênh (cần thiết cho CNN)
    return normalized.reshape(64, 128, 1)

def create_label_mapping(labels):
    # Tìm tất cả các ký tự độc nhất
    unique_chars = set()
    max_length = 0
    
    for label in labels:
        unique_chars.update(label)
        max_length = max(max_length, len(label))
    
    # Tạo mapping
    char_to_index = {char: i+1 for i, char in enumerate(sorted(unique_chars))}
    char_to_index[''] = 0  # Thêm padding token
    
    index_to_char = {i+1: char for i, char in enumerate(sorted(unique_chars))}
    index_to_char[0] = ''  # Thêm padding token
    
    return char_to_index, index_to_char, max_length

def encode_labels(labels, char_to_index, max_length):
    num_samples = len(labels)
    num_classes = len(char_to_index)
    
    # Mã hóa chuỗi thành mảng index
    encoded = np.zeros((num_samples, max_length), dtype=np.int32)
    
    for i, label in enumerate(labels):
        for j, char in enumerate(label):
            if j < max_length:
                encoded[i, j] = char_to_index.get(char, 0)
    
    # Chuyển sang one-hot encoding
    one_hot = np.zeros((num_samples, max_length, num_classes), dtype=np.float32)
    
    for i in range(num_samples):
        for j in range(max_length):
            if encoded[i, j] > 0:
                one_hot[i, j, encoded[i, j]] = 1.0
    
    return one_hot

# ========================= XÂY DỰNG MÔ HÌNH CNN =========================

def build_cnn_model(input_shape, max_length, num_classes):
    # Đầu vào
    inputs = layers.Input(shape=input_shape)
    
    # Khối CNN đầu tiên
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Khối CNN thứ hai
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Khối CNN thứ ba
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Khối CNN thứ tư
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Reshape cho LSTM
    shape = tf.keras.backend.int_shape(x)
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # Bidrectional LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    
    # Thêm một lớp LSTM nữa
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    
    # Đầu ra
    outputs = []
    for i in range(max_length):
        # Một dense layer cho mỗi vị trí trong chuỗi
        output = layers.Dense(num_classes, activation='softmax', name=f'char_{i}')(x[:, i, :])
        outputs.append(output)
    
    # Tổng hợp mô hình
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile mô hình - SỬA LỖI: đảm bảo metrics phù hợp với số lượng đầu ra
    losses = {f'char_{i}': 'categorical_crossentropy' for i in range(max_length)}
    metrics = {f'char_{i}': 'accuracy' for i in range(max_length)}
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    return model

# ========================= HUẤN LUYỆN & ĐÁNH GIÁ =========================

def train_captcha_model(images, labels, char_to_index, max_length, batch_size=32, epochs=50):
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Mã hóa nhãn
    num_classes = len(char_to_index)
    y_train_encoded = encode_labels(y_train, char_to_index, max_length)
    y_val_encoded = encode_labels(y_val, char_to_index, max_length)
    
    # Xây dựng mô hình
    input_shape = images[0].shape
    model = build_cnn_model(input_shape, max_length, num_classes)
    
    # Hiện thông tin mô hình
    model.summary()
    
    # Chuẩn bị dữ liệu đầu ra cho huấn luyện - Định dạng phù hợp với dictionary output
    train_outputs = {f'char_{i}': y_train_encoded[:, i] for i in range(max_length)}
    val_outputs = {f'char_{i}': y_val_encoded[:, i] for i in range(max_length)}
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, train_outputs,
        validation_data=(X_val, val_outputs),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    return model, history

def predict_captcha(model, image, index_to_char, max_length):
    # Thêm dimension batch
    image_batch = np.expand_dims(image, axis=0)
    
    # Dự đoán
    predictions = model.predict(image_batch)
    
    # Giải mã kết quả - với mô hình có multiple outputs
    result = ""
    # Kiểm tra nếu predictions là dictionary
    if isinstance(predictions, dict):
        for i in range(max_length):
            char_pred = predictions[f'char_{i}'][0]
            char_index = np.argmax(char_pred)
            if char_index > 0:  # Bỏ qua padding
                result += index_to_char[char_index]
    # Nếu predictions là list (mô hình cũ)
    else:
        for i in range(max_length):
            char_pred = predictions[i][0]
            char_index = np.argmax(char_pred)
            if char_index > 0:  # Bỏ qua padding
                result += index_to_char[char_index]
    
    return result

# ========================= ĐÁNH GIÁ HIỆU SUẤT =========================

def evaluate_model(model, test_images, test_labels, index_to_char, max_length):
    num_correct = 0
    char_correct = 0
    total_chars = 0
    
    predictions = []
    
    for i, image in enumerate(test_images):
        # Dự đoán
        pred = predict_captcha(model, image, index_to_char, max_length)
        predictions.append(pred)
        
        # So sánh với nhãn thực
        true_label = test_labels[i]
        
        # Tính độ chính xác tổng thể
        if pred == true_label:
            num_correct += 1
        
        # Tính độ chính xác ký tự
        min_len = min(len(pred), len(true_label))
        for j in range(min_len):
            if j < len(pred) and j < len(true_label):
                if pred[j] == true_label[j]:
                    char_correct += 1
        
        total_chars += len(true_label)
    
    # Tính độ chính xác
    overall_accuracy = num_correct / len(test_images)
    char_accuracy = char_correct / total_chars
    
    return overall_accuracy, char_accuracy, predictions

# ========================= PHƯƠNG PHÁP THAY THẾ (NẾU CẦN) =========================

def build_simple_cnn_model(input_shape, max_length, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Đầu ra
    outputs = []
    for i in range(max_length):
        # Sử dụng một dense layer cho mỗi ký tự
        output = layers.Dense(num_classes, activation='softmax', name=f'char_{i}')(x)
        outputs.append(output)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile mô hình - SỬA LỖI: đảm bảo metrics phù hợp với số lượng đầu ra
    losses = {f'char_{i}': 'categorical_crossentropy' for i in range(max_length)}
    metrics = {f'char_{i}': 'accuracy' for i in range(max_length)}
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )
    
    return model

# ========================= CHƯƠNG TRÌNH CHÍNH =========================

def main():
    # Thư mục chứa dữ liệu
    data_folder = "captcha_images"
    
    # Tạo thư mục output
    output_folder = "captcha_model_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load dữ liệu
    print(f"Đang tải dữ liệu từ {data_folder}...")
    images, labels = load_captcha_images(data_folder)
    print(f"Đã tải {len(images)} hình ảnh CAPTCHA")
    
    # Tạo mapping nhãn
    char_to_index, index_to_char, max_length = create_label_mapping(labels)
    print(f"Số ký tự độc nhất: {len(char_to_index)-1}")
    print(f"Độ dài tối đa của CAPTCHA: {max_length}")
    
    # Lưu mapping để sử dụng sau này - SỬA LỖI: thêm encoding="utf-8"
    import json
    with open(os.path.join(output_folder, "char_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({
            "char_to_index": {str(k): v for k, v in char_to_index.items()},
            "index_to_char": {str(k): v for k, v in index_to_char.items()},
            "max_length": max_length
        }, f)
    
    # Huấn luyện mô hình
    print("\nBắt đầu huấn luyện mô hình...")
    try:
        model, history = train_captcha_model(
            images, labels, char_to_index, max_length,
            batch_size=32, epochs=100
        )
        
        # Lưu mô hình
        model.save(os.path.join(output_folder, "captcha_model.h5"))
        print(f"Đã lưu mô hình vào {os.path.join(output_folder, 'captcha_model.h5')}")
        
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình chính: {str(e)}")
        print("Thử với mô hình đơn giản hơn...")
        
        # Thử với mô hình đơn giản hơn
        input_shape = images[0].shape
        model = build_simple_cnn_model(input_shape, max_length, len(char_to_index))
        
        # Chia dữ liệu
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
        
        # Mã hóa nhãn
        y_train_encoded = encode_labels(y_train, char_to_index, max_length)
        y_val_encoded = encode_labels(y_val, char_to_index, max_length)
        
        # Chuẩn bị dữ liệu đầu ra - Định dạng phù hợp với dict output
        train_outputs = {f'char_{i}': y_train_encoded[:, i] for i in range(max_length)}
        val_outputs = {f'char_{i}': y_val_encoded[:, i] for i in range(max_length)}
        
        # Huấn luyện
        history = model.fit(
            X_train, train_outputs,
            validation_data=(X_val, val_outputs),
            batch_size=32, epochs=100,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        # Lưu mô hình
        model.save(os.path.join(output_folder, "captcha_model_simple.h5"))
        print(f"Đã lưu mô hình đơn giản vào {os.path.join(output_folder, 'captcha_model_simple.h5')}")

    # Đánh giá mô hình trên tập kiểm tra
    print("\nĐánh giá mô hình...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    overall_acc, char_acc, predictions = evaluate_model(model, X_test, y_test, index_to_char, max_length)
    
    print(f"Độ chính xác tổng thể: {overall_acc:.4f}")
    print(f"Độ chính xác ký tự: {char_acc:.4f}")
    
    # Lưu kết quả đánh giá - SỬA LỖI: thêm encoding="utf-8"
    with open(os.path.join(output_folder, "evaluation_results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Đánh giá mô hình\n")
        f.write(f"==============\n\n")
        f.write(f"Độ chính xác tổng thể: {overall_acc:.4f}\n")
        f.write(f"Độ chính xác ký tự: {char_acc:.4f}\n\n")
        
        f.write("Chi tiết dự đoán:\n")
        for i, (pred, true) in enumerate(zip(predictions, y_test)):
            f.write(f"{i+1}. Dự đoán: {pred}, Thực tế: {true}\n")
    
    print(f"Đã lưu kết quả đánh giá vào {os.path.join(output_folder, 'evaluation_results.txt')}")
    print("\nHoàn tất quá trình!")

if __name__ == "__main__":
    main()