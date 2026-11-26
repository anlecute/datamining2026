# Chạy project

## Điều kiện đầu vào

- Cài Python 3.10 trở lên và dùng lệnh `py` trong PowerShell.
- File raw phải nằm tại:

  ```text
  All_Beauty.jsonl\All_Beauty.jsonl
  ```

  Đây là file JSONL gốc; không chỉnh sửa hoặc ghi đè file này.

## Các bước thực hiện

1. Cài các thư viện cần thiết:

   ```powershell
   py -m pip install -r requirements.txt
   ```

2. Kiểm tra toàn bộ dữ liệu gốc và tạo bằng chứng audit:

   ```powershell
   py src\audit_raw_data.py
   ```

   File `data/interim/raw_data_audit.json` ghi đường dẫn, kích thước,
   SHA-256 của file raw, số record theo từng quy tắc loại bỏ và phép đối soát
   tổng số dòng. Chạy lại bước này khi file raw thay đổi.

3. Tạo tập dữ liệu đã làm sạch (không làm thay đổi dữ liệu thô):

   ```powershell
   py src\preprocess_all_beauty.py
   ```

   Kết quả gồm `data/interim/cleaned.jsonl` và
   `data/interim/preprocessing_stats.json`.

4. Xuất dữ liệu cho Weka và CSV để kiểm tra:

   ```powershell
   py src\export_arff.py
   ```

   Kết quả gồm `data/processed/amazon_sentiment.arff` và
   `data/processed/amazon_sentiment.csv`.

5. Thực hiện EDA và tạo biểu đồ:

   ```powershell
   py src\eda_all_beauty.py
   ```

   Bước này tạo bảng thống kê tại `outputs/metrics/eda_summary.json`, đoạn
   diễn giải có thể chèn vào report tại `outputs/metrics/eda_interpretation.md`,
   và bốn biểu đồ EDA tại `outputs/figures/`.

6. Huấn luyện bốn baseline trên mẫu ngẫu nhiên 50.000 review:

   ```powershell
   py src\train_models.py
   ```

   Để huấn luyện trên toàn bộ dữ liệu, dùng:

   ```powershell
   py src\train_models.py --max-records 0
   ```

7. Phân tích các dự đoán sai của mô hình tốt nhất:

   ```powershell
   py src\analyze_errors.py
   ```

   Bước này cần chạy sau train. Nó tạo bảng số lỗi theo cặp class và một tập
   ví dụ lỗi để dùng trong phần error analysis của report.

8. Tinh chỉnh có kiểm soát mà không dùng test set để chọn cấu hình:

   ```powershell
   py src\tune_models.py
   ```

   Script dùng validation tách từ train set để chọn cấu hình theo Macro F1. Test
   set 10.000 review chỉ được dùng đúng một lần sau khi cấu hình đã được chọn.

## Kết quả tạo ra

| Vị trí | Nội dung |
|---|---|
| `data/interim/raw_data_audit.json` | Bằng chứng kiểm tra file raw và các quy tắc tiền xử lý. |
| `data/interim/cleaned.jsonl` | Dữ liệu sạch có `review_text` và `sentiment_class`. |
| `data/interim/preprocessing_stats.json` | Thống kê sau tiền xử lý. |
| `data/processed/amazon_sentiment.arff` | Dữ liệu ARFF cho Weka. |
| `data/processed/amazon_sentiment.csv` | Bản CSV để kiểm tra. |
| `outputs/metrics/eda_summary.json` | Thống kê EDA: rating, nhãn và độ dài review. |
| `outputs/metrics/eda_interpretation.md` | Bảng và đoạn diễn giải dùng cho report. |
| `outputs/metrics/test_predictions_best_model.csv` | 10.000 dự đoán trên test set có nhãn thật. |
| `outputs/metrics/error_analysis_report.md` | Tóm tắt các cặp class model dự đoán sai. |
| `outputs/metrics/error_examples.csv` | Ví dụ review dự đoán sai để đọc và phân tích. |
| `outputs/metrics/tuning_results.csv` | Kết quả từng cấu hình trên validation set. |
| `outputs/metrics/tuning_final_test_result.json` | Kết quả cuối trên held-out test set của cấu hình thắng. |
| `outputs/metrics/` | Bảng so sánh mô hình và classification report. |
| `outputs/figures/` | Biểu đồ EDA và confusion matrix của mô hình tốt nhất. |

Script huấn luyện chọn mô hình theo Macro F1. Đây là chỉ số chính vì dữ liệu có
phân bố lớp không cân bằng; không nên chọn mô hình chỉ dựa vào Accuracy.
