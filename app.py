from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import glob
import shutil
from datetime import datetime
import tempfile

app = Flask(__name__)

UNLABELED_FOLDER = 'unlabeled'
CAPTCHA_IMAGES_FOLDER = 'captcha_images'
TEMP_FOLDER = tempfile.mkdtemp()  # Thư mục tạm thời để xử lý ảnh

# Tạo các thư mục cần thiết
for folder in [UNLABELED_FOLDER, CAPTCHA_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Biến lưu trạng thái toàn cục
current_images = []
current_index = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_folder', methods=['POST'])
def select_folder():
    folder_path = request.form.get('folder_path')
    
    if not folder_path or not os.path.isdir(folder_path):
        return render_template('index.html', error="Đường dẫn thư mục không hợp lệ")
    
    # Lấy danh sách ảnh trong thư mục
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_list = []
    
    for ext in image_extensions:
        image_list.extend(glob.glob(os.path.join(folder_path, ext)))
        image_list.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_list:
        return render_template('index.html', error="Không tìm thấy ảnh trong thư mục này")
    
    # Sao chép tất cả ảnh vào thư mục unlabeled
    count = 0
    for img_path in image_list:
        try:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(UNLABELED_FOLDER, filename)
            
            # Nếu ảnh chưa tồn tại trong unlabeled thì sao chép
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                count += 1
        except Exception as e:
            print(f"Error copying {img_path}: {str(e)}")
            
    if count == 0:
        return render_template('index.html', error="Không có ảnh mới để sao chép vào thư mục unlabeled")
            
    # Cập nhật danh sách ảnh hiện tại từ thư mục unlabeled
    update_current_images()
    
    # Chuyển đến ảnh đầu tiên
    return redirect(url_for('process_next_image'))

def update_current_images():
    """Cập nhật danh sách ảnh từ thư mục unlabeled"""
    global current_images, current_index
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_list = []
    
    for ext in image_extensions:
        image_list.extend(glob.glob(os.path.join(UNLABELED_FOLDER, ext)))
        image_list.extend(glob.glob(os.path.join(UNLABELED_FOLDER, ext.upper())))
    
    current_images = image_list
    current_index = 0 if image_list else -1

@app.route('/process_next_image')
def process_next_image():
    try:
        global current_images, current_index
        
        # Cập nhật danh sách ảnh hiện tại
        update_current_images()
        
        if not current_images:
            return render_template('complete.html', message="Tất cả ảnh đã được xử lý")
        
        if current_index >= len(current_images):
            current_index = 0
        
        if current_images:
            current_image_path = current_images[current_index]
            current_index += 1
            
            # Xử lý ảnh
            return process_image(current_image_path)
        else:
            return render_template('complete.html', message="Không tìm thấy ảnh chưa xử lý")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=f"Lỗi khi xử lý ảnh tiếp theo: {str(e)}")

def process_image(image_path):
    try:
        # Lấy tên file gốc
        base_filename = os.path.basename(image_path)
        
        # Đường dẫn tệp tạm thời
        temp_path = os.path.join(TEMP_FOLDER, base_filename)
        temp_processed_path = os.path.join(TEMP_FOLDER, f"processed_{base_filename}")
        temp_top_path = os.path.join(TEMP_FOLDER, f"top_{base_filename}")
        temp_top_display_path = os.path.join(TEMP_FOLDER, f"top_display_{base_filename}")
        
        # Sao chép file ảnh vào thư mục tạm thời để xử lý
        shutil.copy2(image_path, temp_path)
        
        # Đọc ảnh với OpenCV
        image = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return render_template('index.html', error="Không thể đọc ảnh. Vui lòng kiểm tra định dạng ảnh.")
        
        # Kiểm tra xem ảnh có kênh alpha không, nếu không thì thêm vào
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Thêm kênh alpha (trong suốt)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Chuyển đổi ảnh sang HSV để làm việc tốt hơn với màu sắc
        # Chỉ sử dụng 3 kênh đầu tiên cho HSV
        hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
        
        # Xác định vùng màu xám
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([180, 30, 220])
        
        # Tạo mặt nạ cho các vùng màu xám
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Tạo mặt nạ cho vùng không phải màu xám 
        non_gray_mask = cv2.bitwise_not(gray_mask)
        
        # Tạo ảnh đã xử lý với nền trong suốt
        processed_image = image.copy()
        
        # 1. Đặt kênh alpha = 0 (trong suốt) cho các khu vực màu xám trong ảnh chính
        processed_image[:, :, 3] = np.where(gray_mask == 255, 0, processed_image[:, :, 3])
        
        # 2. Đặt tất cả màu sắc khác thành màu đen (giữ nguyên kênh alpha)
        visible_non_gray_mask = np.logical_and(non_gray_mask == 255, processed_image[:, :, 3] > 0)
        processed_image[visible_non_gray_mask, 0] = 0  # B
        processed_image[visible_non_gray_mask, 1] = 0  # G
        processed_image[visible_non_gray_mask, 2] = 0  # R
        
        # Tạo ảnh cuối cùng với nền trắng cho hiển thị web
        white_bg_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        
        # Đặt chỗ có mask không trong suốt thành màu đen trên nền trắng
        for c in range(0, 3):
            white_bg_image[:, :, c] = np.where(processed_image[:, :, 3] > 0, processed_image[:, :, c], white_bg_image[:, :, c])
        
        # Lưu ảnh đã xử lý với nền trong suốt vào thư mục tạm thời
        cv2.imwrite(temp_processed_path, processed_image)
        
        # ---------------------------------
        # CẮT PHẦN TRÊN CỦA ẢNH (CHỮ)
        # ---------------------------------
        
        # Cắt ảnh theo tọa độ cố định
        # Phần trên: x=0, y=0, w=128, h=32
        
        # Đảm bảo ảnh đủ lớn để cắt
        height, width = processed_image.shape[:2]
        top_w, top_h = 128, 32
        top_x, top_y = 0, 0
        
        # Điều chỉnh kích thước nếu cần
        top_w = min(top_w, width)
        top_h = min(top_h, height)
        
        # Cắt phần trên
        if top_y + top_h <= height and top_x + top_w <= width:
            top_image = processed_image[top_y:top_y+top_h, top_x:top_x+top_w].copy()
            cv2.imwrite(temp_top_path, top_image)
            
            # Tạo phiên bản hiển thị cho phần trên
            top_white_bg = np.ones((top_h, top_w, 3), dtype=np.uint8) * 255
            for c in range(0, 3):
                top_white_bg[:, :, c] = np.where(top_image[:, :, 3] > 0, top_image[:, :, c], top_white_bg[:, :, c])
            cv2.imwrite(temp_top_display_path, top_white_bg)
        else:
            # Nếu không thể cắt theo tọa độ yêu cầu, để trống phần trên
            top_white_bg = np.ones((32, 128, 3), dtype=np.uint8) * 255
            cv2.imwrite(temp_top_display_path, top_white_bg)
            cv2.imwrite(temp_top_path, np.zeros((32, 128, 4), dtype=np.uint8))
        
        # Lấy tiến độ xử lý ảnh
        progress = {
            'current': current_index,
            'total': len(current_images)
        }
        
        # Đảm bảo đường dẫn hợp lệ cho template
        input_image_url = f"/image/{base_filename}?type=original"
        output_top_url = f"/image/{base_filename}?type=top_display"
        
        # Trích xuất tên file gốc (không có phần mở rộng) làm giá trị gợi ý cho top_text
        suggested_text = os.path.splitext(base_filename)[0]
        
        # Trả về trang với thông tin kết quả và form để nhập dữ liệu
        return render_template(
            'training.html',
            id=base_filename,
            input_image=input_image_url,
            output_top=output_top_url,
            current_index=progress['current'],
            total_images=progress['total'],
            suggested_text=suggested_text
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Lỗi xử lý ảnh: {str(e)}")
@app.route('/import_labeled_images', methods=['POST'])
def import_labeled_images():
    try:
        folder_path = request.form.get('folder_path')
        
        if not folder_path or not os.path.isdir(folder_path):
            return render_template('index.html', error="Đường dẫn thư mục không hợp lệ")
        
        # Lấy danh sách ảnh trong thư mục
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_list = []
        
        for ext in image_extensions:
            image_list.extend(glob.glob(os.path.join(folder_path, ext)))
            image_list.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_list:
            return render_template('index.html', error="Không tìm thấy ảnh trong thư mục này")
        
        # Số lượng ảnh đã nhập thành công
        imported_count = 0
        # Số lượng ảnh bị bỏ qua (do trùng lặp)
        skipped_count = 0
        
        for img_path in image_list:
            try:
                # Lấy tên file gốc
                base_filename = os.path.basename(img_path)
                
                # Đường dẫn đích trong thư mục captcha_images
                captcha_path = os.path.join(CAPTCHA_IMAGES_FOLDER, base_filename)
                
                # Kiểm tra xem file đã tồn tại trong thư mục captcha_images chưa
                if os.path.exists(captcha_path):
                    skipped_count += 1
                    continue
                
                # Copy trực tiếp file vào thư mục captcha_images
                shutil.copy2(img_path, captcha_path)
                imported_count += 1
                
            except Exception as e:
                print(f"Error importing {img_path}: {str(e)}")
        
        # Hiển thị thông báo kết quả
        message = f"Đã nhập thành công {imported_count} ảnh. Bỏ qua {skipped_count} ảnh trùng lặp."
        
        # Trả về trang kết quả
        return render_template('import_result.html', 
                               message=message, 
                               imported_count=imported_count, 
                               skipped_count=skipped_count,
                               total_count=len(image_list))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Lỗi nhập ảnh: {str(e)}")
@app.route('/save_training', methods=['POST'])
def save_training():
    try:
        image_id = request.form.get('image_id')
        top_text = request.form.get('top_text', '')
        
        # Đường dẫn ảnh gốc trong thư mục unlabeled
        unlabeled_path = os.path.join(UNLABELED_FOLDER, image_id)
        
        # Đường dẫn ảnh đã xử lý trong thư mục tạm thời
        temp_top_path = os.path.join(TEMP_FOLDER, f"top_{image_id}")
        
        # Tạo tên file mới dựa trên top_text
        file_ext = os.path.splitext(image_id)[1]
        new_filename = f"{top_text}{file_ext}"
        captcha_path = os.path.join(CAPTCHA_IMAGES_FOLDER, new_filename)
        
        # Sao chép ảnh đã xử lý vào thư mục captcha_images với tên mới
        if os.path.exists(temp_top_path):
            shutil.copy2(temp_top_path, captcha_path)
            print(f"Đã lưu ảnh đã xử lý: {captcha_path}")
            
            # Xóa ảnh gốc khỏi thư mục unlabeled
            if os.path.exists(unlabeled_path):
                os.remove(unlabeled_path)
                print(f"Đã xóa ảnh gốc: {unlabeled_path}")
        
        # Cập nhật danh sách ảnh hiện tại
        update_current_images()
        
        # Chuyển đến ảnh tiếp theo
        return redirect(url_for('process_next_image'))
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/image/<path:filename>')
def serve_image(filename):
    image_type = request.args.get('type', 'original')
    
    if image_type == 'original':
        # Tìm ảnh từ unlabeled folder
        file_path = os.path.join(UNLABELED_FOLDER, filename)
    elif image_type == 'top_display':
        # Tìm ảnh từ thư mục tạm thời
        file_path = os.path.join(TEMP_FOLDER, f"top_display_{filename}")
    else:
        # Mặc định
        file_path = os.path.join(UNLABELED_FOLDER, filename)
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        
        # Trả về ảnh placeholder nếu file không tồn tại
        placeholder_path = os.path.join(os.path.dirname(__file__), 'static', 'placeholder.png')
        
        # Nếu không có sẵn placeholder, tạo một ảnh trống
        if not os.path.exists(placeholder_path):
            # Đảm bảo thư mục static tồn tại
            os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
            
            # Tạo một ảnh placeholder đơn giản
            placeholder = np.ones((32, 128, 3), dtype=np.uint8) * 240  # Màu xám nhạt
            
            # Thêm text "No Image"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(placeholder, "No Image", (10, 20), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
            
            # Lưu ảnh placeholder
            cv2.imwrite(placeholder_path, placeholder)
        
        print(f"File not found: {file_path}, returning placeholder")
        return send_file(placeholder_path)
    
    try:
        print(f"Serving file: {file_path}")
        return send_file(file_path)
    except Exception as e:
        print(f"Error serving file {file_path}: {str(e)}")
        placeholder_path = os.path.join(os.path.dirname(__file__), 'static', 'placeholder.png')
        return send_file(placeholder_path)

@app.route('/captcha_images/<path:filename>')
def captcha_image(filename):
    file_path = os.path.join(CAPTCHA_IMAGES_FOLDER, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    
    # If file doesn't exist, serve placeholder
    placeholder_path = os.path.join(os.path.dirname(__file__), 'static', 'placeholder.png')
    
    # Create placeholder if it doesn't exist
    if not os.path.exists(placeholder_path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
        
        # Create placeholder image
        placeholder = np.ones((32, 128, 3), dtype=np.uint8) * 240
        cv2.putText(placeholder, "No Image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        # Make sure to save the image to a file instead of returning the array
        cv2.imwrite(placeholder_path, placeholder)
    
    # Must return a Flask response object, not a NumPy array
    return send_file(placeholder_path)

@app.route('/view_captcha_images')
def view_captcha_images():
    # Lấy danh sách các ảnh trong thư mục captcha_images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_list = []
    
    for ext in image_extensions:
        image_list.extend(glob.glob(os.path.join(CAPTCHA_IMAGES_FOLDER, ext)))
        image_list.extend(glob.glob(os.path.join(CAPTCHA_IMAGES_FOLDER, ext.upper())))
    
    # Chuyển đổi đường dẫn đầy đủ thành tên file
    image_names = [os.path.basename(img) for img in image_list]
    
    return render_template('view_captcha.html', images=image_names)

@app.route('/view_unlabeled')
def view_unlabeled():
    # Lấy danh sách các ảnh trong thư mục unlabeled
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_list = []
    
    for ext in image_extensions:
        image_list.extend(glob.glob(os.path.join(UNLABELED_FOLDER, ext)))
        image_list.extend(glob.glob(os.path.join(UNLABELED_FOLDER, ext.upper())))
    
    # Chuyển đổi đường dẫn đầy đủ thành tên file
    image_names = [os.path.basename(img) for img in image_list]
    
    return render_template('view_unlabeled.html', images=image_names, count=len(image_names))

if __name__ == '__main__':
    app.run(debug=True)