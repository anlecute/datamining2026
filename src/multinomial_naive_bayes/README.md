# Multinomial Naive Bayes + Weka

Pipeline gồm chuyển JSONL đã làm sạch sang ARFF, train `StringToWordVector` + `NaiveBayesMultinomial`, lưu model và đánh giá stratified 10-fold CV. Chạy các lệnh dưới đây từ **thư mục gốc của project** (thư mục chứa `src` và `data`).

## 1. Chuẩn bị

Cần JDK 11 trở lên (`java`, `javac` trong PATH). Trong PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\src\multinomial_naive_bayes\setup.ps1
```

Nếu có Weka tại `C:\Program Files\Weka-3-9-6\weka.jar`, script và `run.bat` dùng Weka đó. Nếu không, setup tải Weka 3.8.6 và các dependency từ Maven Central, cần kết nối internet. Dùng cùng phiên bản Weka để train và load model.

## 2. Chuyển dữ liệu

```powershell
.\src\multinomial_naive_bayes\run.bat convert ".\data\interim\cleaned_minimal.jsonl" ".\data\processed\amazon_mnb.arff"
```

Input là **JSONL đã làm sạch và gán nhãn**, không phải raw review. Mỗi dòng cần `sentiment_class` thuộc Negative, Neutral, Positive và `review_text`; nếu thiếu `review_text`, chương trình ghép `title` và `text`. Nó không tự lọc duplicate hay tạo label từ rating. Sửa đường dẫn theo vị trí dữ liệu thực tế.

## 3. Train và lưu model

```powershell
.\src\multinomial_naive_bayes\run.bat train ".\data\processed\amazon_mnb.arff" ".\outputs\models\mnb.model"
```

Model được train trên toàn bộ ARFF đã cung cấp, gồm cả filter và classifier. Baseline dùng lowercase, unigram, tần suất từ (word counts), `minTermFreq=2`, `wordsToKeep=20000` trên toàn bộ lớp; không bật TF-IDF, stemming hoặc loại stop words. `wordsToKeep` là giới hạn gần đúng theo cách Weka xử lý các từ có cùng tần suất. Sửa `MnbFactory.java` để thử cấu hình khác, rồi train lại.

## 4. Đánh giá

```powershell
.\src\multinomial_naive_bayes\run.bat evaluate ".\data\processed\amazon_mnb.arff" ".\outputs\models\mnb.model" ".\outputs\metrics\mnb_cv.json"
```

Evaluation load **cấu hình** từ model đã lưu, sau đó tạo filter và classifier mới, train lại trên mỗi training fold và đánh giá fold còn lại. Model train trên toàn bộ dữ liệu không được dùng trực tiếp để dự đoán fold, tránh leakage. Dữ liệu được shuffle với seed 42 và stratify. Mỗi lớp cần ít nhất 10 mẫu. Kết quả JSON được cập nhật sau mỗi fold (`in_progress`) và có trạng thái `complete` sau fold thứ 10.

- Accuracy và macro precision, macro recall, macro F1 có thang 0–1; trường hợp không dự đoán một lớp thì precision của lớp đó bằng 0.
- MAE/RMSE tính từ nhãn dự đoán theo thứ tự Negative=0, Neutral=1, Positive=2 để so sánh với pipeline SVM trong project. Đây là **ordinal label error**, không phải MAE/RMSE theo xác suất mặc định của Weka.
- `metrics_mean` là trung bình 10 fold; `metrics_std` là độ lệch chuẩn mẫu (chia cho 9). Thời gian fit gồm vectorization và huấn luyện. `elapsed_seconds` đo từ lúc bắt đầu vòng CV sau khi load dữ liệu/model.
- CV trên cùng dữ liệu đã dùng train model là đánh giá lại cấu hình bằng các model mới, không phải đánh giá riêng model cuối trên một test set độc lập. Nếu chọn/tinh chỉnh cấu hình qua CV, nên giữ test set riêng cho đánh giá cuối.

## 5. Lưu ý tài nguyên

ARFF, model và báo cáo được tạo khi chạy, không có sẵn kết quả thực nghiệm thật. Chương trình load ARFF vào RAM và train lại 10 lần. Bắt đầu với dữ liệu nhỏ có ít nhất 10 record mỗi lớp; tập 694.253 reviews có thể cần nhiều RAM/thời gian. Heap mặc định là `-Xmx8g` trong `run.bat`, điều chỉnh theo máy. Nếu JSONL trong project chỉ chứa `version https://git-lfs.github.com/spec/v1`, đó là Git LFS pointer; cần lấy file dữ liệu thực trước khi chạy.
