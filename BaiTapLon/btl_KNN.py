import cv2
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Đọc dữ liệu màu từ file CSV
color_data = pd.read_csv('colors2.csv')

# Hàm tính toán màu gần nhất
def get_closest_color(r, g, b, color_data):
    distances = np.sqrt(
        (color_data['R'] - r) ** 2 +
        (color_data['G'] - g) ** 2 +
        (color_data['B'] - b) ** 2
    )
    return color_data.loc[distances.idxmin(), 'Color Name']

# Hàm xử lý ảnh với kNN
def process_image_with_knn(image_path, n_colors=5, k=3):
    # Đọc ảnh và chuyển sang không gian màu RGB
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize ảnh để tăng tốc độ xử lý
    small_image = cv2.resize(image_rgb, (50, 50))
    pixels = small_image.reshape(-1, 3)

    # KMeans để tạo nhãn ban đầu
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Huấn luyện kNN với dữ liệu từ KMeans
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(pixels, labels)

    # Dự đoán nhãn mới cho tất cả các pixel bằng kNN
    predictions = knn.predict(pixels)
    label_reshaped = predictions.reshape(50, 50)  # Chuyển thành ma trận để dễ xử lý

    return image, dominant_colors, label_reshaped

# Hàm hiển thị các cụm màu trên nền ảnh gốc
def display_color_clusters_with_original_background(image, dominant_colors, label_reshaped):
    h, w, _ = image.shape
    label_map = cv2.resize(label_reshaped.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)

    for i, color in enumerate(dominant_colors):
        mask = label_map == i
        mask_uint8 = mask.astype('uint8') * 255

        # Tạo hình ảnh cụm màu
        cluster_image = np.zeros_like(image)
        cluster_image[mask] = image[mask]

        # Lấy tên màu
        r, g, b = color
        color_name = get_closest_color(r, g, b, color_data)

        # Hiển thị tên màu trên ảnh
        cv2.putText(
            cluster_image,
            f"Color: {color_name}",
            (10, 30),  # Vị trí (x, y) - Góc trên bên trái
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,  # Kích thước font chữ
            (255,0,0),  # Màu chữ: Đỏ (BGR format)
            2,  # Độ dày nét chữ
            cv2.LINE_AA
        )

        # Hiển thị ảnh cụm màu kèm tên
        cv2.imshow(f"Cụm màu {i + 1}", cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Đánh giá hiệu suất
def evaluate_clustering(ground_truth, predictions):
    print("=== Đánh giá hiệu suất kNN ===")
    print(f"Accuracy: {accuracy_score(ground_truth, predictions):.2f}")
    print(f"Precision: {precision_score(ground_truth, predictions, average='weighted'):.2f}")
    print(f"Recall: {recall_score(ground_truth, predictions, average='weighted'):.2f}")
    print(f"F1-Score: {f1_score(ground_truth, predictions, average='weighted'):.2f}")
    print("\n=== Báo cáo chi tiết ===")
    print(classification_report(ground_truth, predictions))

# Chương trình chính
if __name__ == "__main__":
    Tk().withdraw()
    image_path = askopenfilename(title="Chọn file ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if image_path:
        # Xử lý ảnh với kNN
        image, dominant_colors, label_reshaped = process_image_with_knn(image_path)

        # Tạo nhãn giả lập để đánh giá
        ground_truth = np.random.randint(0, len(dominant_colors), size=(50, 50)).flatten()
        predictions = label_reshaped.flatten()

        # Đánh giá hiệu suất
        evaluate_clustering(ground_truth, predictions)

        # Hiển thị các cụm màu
        display_color_clusters_with_original_background(image, dominant_colors, label_reshaped)
        print("Đã hiển thị tất cả các cụm màu!")
    else:
        print("Bạn chưa chọn file ảnh!")
