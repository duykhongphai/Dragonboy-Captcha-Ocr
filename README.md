# 🔍 CAPTCHA Processor

<div align="center">
  
  ![CAPTCHA Processor Banner](https://via.placeholder.com/800x200/4361ee/ffffff?text=CAPTCHA+Processor)
  
  [![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
  [![Flask](https://img.shields.io/badge/flask-2.0.1-brightgreen.svg)](https://flask.palletsprojects.com/)
  [![OpenCV](https://img.shields.io/badge/opencv-4.5.3-orange.svg)](https://opencv.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  
  **Ứng dụng xử lý và quản lý dữ liệu CAPTCHA cho mô hình học máy**
  
</div>

## ✨ Tính năng chính

<table>
  <tr>
    <td width="50%">
      <h3>🖼️ Xử lý ảnh thông minh</h3>
      <ul>
        <li>Tự động lọc nhiễu và nền</li>
        <li>Tách phần chữ cái khỏi hình ảnh</li>
        <li>Chuẩn hóa ảnh cho học máy</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📋 Giao diện nhập liệu</h3>
      <ul>
        <li>Giao diện người dùng trực quan</li>
        <li>Xem ảnh gốc và ảnh đã xử lý</li>
        <li>Các phím tắt tiện lợi</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🗃️ Quản lý dữ liệu</h3>
      <ul>
        <li>Hiển thị ảnh chưa xử lý</li>
        <li>Hiển thị ảnh đã xử lý</li>
        <li>Thống kê tiến độ xử lý</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📤 Nhập hàng loạt</h3>
      <ul>
        <li>Nhập nhiều ảnh đã gán nhãn</li>
        <li>Sao chép tự động</li>
        <li>Báo cáo kết quả chi tiết</li>
      </ul>
    </td>
  </tr>
</table>

## 🚀 Bắt đầu nhanh

### Yêu cầu hệ thống

- Python 3.6+
- Các thư viện: Flask, OpenCV, NumPy

### Cài đặt

```bash
# Clone repository
git clone https://github.com/duykhongphai/Dragonboy-Captcha-Ocr.git
cd Dragonboy-Captcha-Ocr

# Cài đặt thư viện yêu cầu
pip install -r requirements.txt

# Chạy ứng dụng
python app.py
```

Sau khi khởi chạy, mở trình duyệt và truy cập: `http://localhost:5000`

## 📋 Hướng dẫn sử dụng

<details>
<summary><b>Xử lý ảnh thủ công</b></summary>
<br>

### Xử lý từng ảnh CAPTCHA và nhập dữ liệu

1. Từ trang chủ, chọn **Xử lý ảnh thủ công**
2. Nhập đường dẫn thư mục chứa ảnh CAPTCHA
3. Nhấn **Bắt đầu xử lý ảnh**
4. Với mỗi ảnh được hiển thị:
   - Quan sát cả ảnh gốc và ảnh đã xử lý
   - Nhập nội dung CAPTCHA vào ô văn bản
   - Nhấn **Enter** hoặc **Lưu & Tiếp tục** để chuyển ảnh tiếp theo

> **Mẹo:** Sử dụng phím **Enter** để di chuyển nhanh qua các ảnh

</details>

<details>
<summary><b>Nhập ảnh đã có sẵn nhãn</b></summary>
<br>

### Nhập nhiều ảnh đã được gán nhãn

1. Từ trang chủ, chọn **Nhập ảnh đã có sẵn nhãn**
2. Nhập đường dẫn thư mục chứa ảnh đã gán nhãn
3. Nhấn **Nhập ảnh đã gán nhãn**
4. Xem báo cáo kết quả nhập:
   - Số ảnh đã nhập thành công
   - Số ảnh bị bỏ qua (trùng lặp)
   - Tổng số ảnh được xử lý

> **Lưu ý:** Tên file (không có phần mở rộng) được sử dụng làm nhãn của ảnh. Ví dụ: `ABCDEF.png` sẽ có nhãn là "ABCDEF"

</details>

<details>
<summary><b>Huấn luyện và thử nghiệm mô hình</b></summary>
<br>

### Huấn luyện mô hình nhận dạng CAPTCHA

Sau khi đã có đủ dữ liệu trong thư mục `captcha_images`, bạn có thể huấn luyện mô hình nhận dạng:

```bash
# Chạy script huấn luyện
python text_recognition.py --train
```

Script sẽ:
1. Đọc tất cả ảnh từ thư mục `captcha_images`
2. Tách dữ liệu thành tập huấn luyện và tập kiểm tra
3. Huấn luyện mô hình CNN để nhận dạng ký tự
4. Lưu mô hình đã huấn luyện vào `model.h5`

### Thử nghiệm mô hình

Để kiểm tra độ chính xác của mô hình với một ảnh cụ thể:

```bash
# Thử nghiệm trên một ảnh
python text_recognition.py --test path/to/image.png
```

Để đánh giá mô hình trên toàn bộ tập dữ liệu kiểm tra:

```bash
# Đánh giá trên tập kiểm tra
python text_recognition.py --evaluate
```

Kết quả sẽ hiển thị:
- Độ chính xác tổng thể
- Các trường hợp nhận dạng sai
- Ma trận nhầm lẫn

> **Mẹo:** Tăng số lượng ảnh huấn luyện để cải thiện độ chính xác. Ít nhất 1000 ảnh cho kết quả tốt.

</details>

## 📁 Cấu trúc thư mục

```
captcha-processor/
├── app.py                 # File chính của ứng dụng
├── text_recognition.py    # Script huấn luyện & nhận dạng
├── requirements.txt       # Danh sách thư viện yêu cầu
├── README.md              # File hướng dẫn này
├── templates/             # Các template HTML
│   ├── index.html         # Trang chủ
│   ├── training.html      # Trang nhập dữ liệu
│   ├── view_captcha.html  # Xem ảnh đã xử lý
│   ├── view_unlabeled.html # Xem ảnh chưa xử lý
│   └── import_result.html # Kết quả nhập ảnh
├── static/                # File tĩnh (CSS, JS, hình ảnh)
├── model.h5               # Mô hình đã huấn luyện
├── unlabeled/             # Thư mục chứa ảnh chưa xử lý
└── captcha_images/        # Thư mục chứa ảnh đã xử lý
```

## 💡 Xử lý sự cố

<table>
  <tr>
    <th>Vấn đề</th>
    <th>Giải pháp</th>
  </tr>
  <tr>
    <td>Không tìm thấy thư mục</td>
    <td>Sử dụng đường dẫn tuyệt đối đến thư mục. Ví dụ: <code>C:\CaptchaImages</code> hoặc <code>/home/user/captcha_images</code></td>
  </tr>
  <tr>
    <td>Ảnh không được xử lý đúng</td>
    <td>Đảm bảo ảnh có định dạng phổ biến (jpg, png, bmp). Kiểm tra kích thước và chất lượng ảnh</td>
  </tr>
  <tr>
    <td>Lỗi khi chạy ứng dụng</td>
    <td>Kiểm tra đã cài đầy đủ các thư viện trong <code>requirements.txt</code>. Chạy với quyền admin nếu cần</td>
  </tr>
  <tr>
    <td>Ảnh không hiển thị</td>
    <td>Kiểm tra quyền truy cập thư mục và đảm bảo ứng dụng web có quyền đọc/ghi file</td>
  </tr>
</table>

## 🛠️ Công nghệ sử dụng

- **Flask**: Framework web nhẹ và linh hoạt
- **OpenCV**: Thư viện xử lý ảnh mạnh mẽ
- **NumPy**: Thư viện tính toán số học
- **HTML/CSS**: Giao diện người dùng
- **JavaScript**: Tương tác phía client

## 📄 Giấy phép

Dự án này được phân phối dưới [Giấy phép MIT](https://opensource.org/licenses/MIT).

---

<div align="center">
  
  🌟 **Đóng góp và phản hồi luôn được chào đón!** 🌟
  
  [Báo cáo lỗi](https://github.com/duykhongphai/Dragonboy-Captcha-Ocr/issues) | [Đóng góp](https://github.com/duykhongphai/Dragonboy-Captcha-Ocr/pulls)
  
</div>
