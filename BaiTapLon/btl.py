import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tkinter import Tk, Button, filedialog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Đọc dữ liệu màu từ file CSV
color_data = pd.read_csv('colors2.csv')
colors_data = pd.read_csv('colors.csv')

# Hàm tính toán màu gần nhất
def get_closest_color(r, g, b, color_data):
    distances = np.sqrt(
        (color_data['R'] - r) ** 2 +
        (color_data['G'] - g) ** 2 +
        (color_data['B'] - b) ** 2
    )
    return color_data.loc[distances.idxmin(), 'Color Name']

def get_color_name(h, s, v):
    min_diff = float('inf')
    color_name = ""
    for i in range(len(colors_data)):
        row = colors_data.loc[i]
        diff = np.sqrt((h - row['H']) ** 2 + (s - row['S']) ** 2 + (v - row['V']) ** 2)
        if diff < min_diff:
            min_diff = diff
            color_name = row['ColorName']
    return color_name
# Hàm xử lý ảnh và trích xuất màu sắc chính
def process_image(image, n_colors=5):
    #Chuyển đổi ảnh từ BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Resize ảnh về kích thước nhỏ hơn (50x50) để giảm độ phức tạp tính toán
    small_image = cv2.resize(image_rgb, (50, 50))
    pixels = small_image.reshape(-1, 3)

    #Chia ảnh thành n_colors cụm màu bằng KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42) #Số lượng cụm cần phân chia (tương ứng với số màu chính cần tìm),Đảm bảo kết quả ổn định khi chạy lại.
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    label_reshaped = labels.reshape(50, 50)

    return dominant_colors, label_reshaped



# Hàm hiển thị các cụm màu trên nền ảnh gốc
def display_color_clusters(image, dominant_colors, label_reshaped):
    h, w, _ = image.shape
    label_map = cv2.resize(label_reshaped.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)

    # Tạo một bản sao của ảnh gốc để hiển thị kết quả
    clustered_image = np.zeros_like(image)

    # Tô màu từng cụm và hiển thị tên màu
    for i, color in enumerate(dominant_colors):
        mask = label_map == i
        cluster_image = np.zeros_like(image)
        cluster_image[mask] = image[mask]  # Áp dụng màu cụm vào các pixel thuộc cụm đó

        # Lấy tên màu
        r, g, b = color
        color_name = get_closest_color(r, g, b, color_data)

        # Hiển thị tên màu góc trên bên trái của ảnh cụm
        cv2.putText(
            cluster_image,
            f"Mau: {color_name}",
            (10, 30),  # Vị trí hiển thị (x, y) - Góc trên bên trái
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Kích thước font chữ
            (255,0,0),  # Màu chữ: Trắng
            2,  # Độ dày nét chữ
            cv2.LINE_AA
        )

        # Hiển thị ảnh cụm màu kèm tên màu
        window_name = f"Cum Mau {i + 1}"
        cv2.imshow(window_name, cv2.cvtColor(cluster_image, cv2.COLOR_BGR2RGB))

    # Đánh giá phân cụm ngay khi hiển thị
    predictions = label_map.flatten()
    ground_truth = np.random.randint(0, len(dominant_colors), size=predictions.size)  # Dữ liệu ngẫu nhiên thay thế ground truth thực tế
    evaluate_clustering(ground_truth, predictions)



# Hàm đánh giá phân cụm
def evaluate_clustering(ground_truth, predictions):
    print("=== Đánh giá hiệu suất phân cụm Kmeans===")

    try:
        print(f"Accuracy: {accuracy_score(ground_truth, predictions):.2f}")
        print(f"Precision: {precision_score(ground_truth, predictions, average='weighted', zero_division=1):.2f}")
        print(f"Recall: {recall_score(ground_truth, predictions, average='weighted', zero_division=1):.2f}")
        print(f"F1-Score: {f1_score(ground_truth, predictions, average='weighted', zero_division=1):.2f}")

        # Báo cáo chi tiết
        print("\n=== Báo cáo chi tiết ===")
        print(classification_report(ground_truth, predictions, zero_division=1))
    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình: {e}")


def open_camera():
    # Mở camera (0 là camera mặc định)
    cap = cv2.VideoCapture(0)

    # Thiết lập độ phân giải cho khung hình (rộng 1280, cao 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Đọc khung hình từ camera
        _, frame = cap.read()

        # Áp dụng Gaussian Blur để làm giảm nhiễu
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Chuyển đổi không gian màu từ BGR sang HSV
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Lấy kích thước của khung hình
        height, width, _ = frame.shape

        # Tính tọa độ trung tâm khung hình
        cx, cy = width // 2, height // 2  # (cx, cy) là tọa độ tâm

        # Tính trung bình giá trị HSV trong vùng 5x5 tại trung tâm
        region = hsv_frame[cy - 2:cy + 3, cx - 2:cx + 3]  # Vùng 5x5 quanh tâm
        avg_h = np.mean(region[:, :, 0])  # Giá trị trung bình Hue (H)
        avg_s = np.mean(region[:, :, 1])  # Giá trị trung bình Saturation (S)
        avg_v = np.mean(region[:, :, 2])  # Giá trị trung bình Value (V)

        # Chuẩn hóa giá trị sáng (V) để đảm bảo kết quả nhất quán
        avg_v = min(255, max(0, avg_v))  # Giới hạn giá trị trong khoảng [0, 255]

        # Xác định tên màu sắc dựa trên giá trị HSV trung bình
        color_name = get_color_name(avg_h, avg_s, avg_v)

        # Hiển thị tên màu trên khung hình
        cv2.putText(
            frame,  # Khung hình
            f"Color: {color_name}",  # Nội dung hiển thị
            (10, 70),  # Tọa độ hiển thị (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font chữ
            1.5,  # Kích thước chữ
            (255, 255, 255),  # Màu chữ (trắng)
            2  # Độ dày của nét chữ
        )

        # Vẽ một vòng tròn nhỏ tại tâm khung hình
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 3)

        # Hiển thị khung hình trong cửa sổ "Camera"
        cv2.imshow("Camera", frame)

        # Kiểm tra phím nhấn, nhấn 'ESC' (mã ASCII là 27) để thoát
        if cv2.waitKey(1) == 27:
            break

    # Giải phóng tài nguyên camera
    cap.release()
    # Đóng tất cả các cửa sổ hiển thị
    cv2.destroyAllWindows()




# Chọn ảnh từ file
def select_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn file ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        image = cv2.imread(file_path)
        dominant_colors, label_reshaped = process_image(image)
        display_color_clusters(image, dominant_colors, label_reshaped)

    else:
        print("Không có ảnh nào được chọn!")



# Giao diện menu
def create_menu():
    root = Tk()
    root.title("Phân Cụm Màu Chính")
    root.geometry("300x200")

    # Nút mở camera
    btn_camera = Button(
        root, text="Mở Camera", command=open_camera, height=2, width=20, bg="lightblue"
    )
    btn_camera.pack(pady=10)

    # Nút chọn ảnh từ folder
    btn_folder = Button(
        root, text="Chọn Ảnh Từ File", command=select_image, height=2, width=20, bg="lightgreen"
    )
    btn_folder.pack(pady=10)

    # Nút thoát
    btn_exit = Button(
        root, text="Thoát", command=root.destroy, height=2, width=20, bg="red"
    )
    btn_exit.pack(pady=10)

    root.mainloop()



# Chạy chương trình
if __name__ == "__main__":
    create_menu()
