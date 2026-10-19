# Report Phase 1 — Khảo sát và trực quan hóa dữ liệu

## Project: Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu Phase 1

Phase 1 thực hiện Exploratory Data Analysis (EDA) trên dataset sau làm sạch.
Mục tiêu là mô tả đặc điểm của dữ liệu bằng số liệu và biểu đồ trước khi phân
tích kết quả mô hình. Các câu hỏi cần trả lời là:

1. Có bao nhiêu record raw và bao nhiêu record sạch?
2. Record bị loại vì lý do gì?
3. Rating nào xuất hiện nhiều nhất?
4. Phân bố Negative, Neutral và Positive có mất cân bằng không?
5. Review trung bình dài bao nhiêu từ?
6. Độ dài review có khác nhau giữa ba lớp cảm xúc không?

## 2. Dữ liệu sử dụng

Dataset là file User Reviews của category `All_Beauty` trong Amazon Reviews
2023. File raw được giữ nguyên tại:

```text
C:\AnLe\datamining\All_Beauty.jsonl\All_Beauty.jsonl
```

EDA được thực hiện trên file dữ liệu đã làm sạch:

```text
C:\AnLe\datamining\data\interim\cleaned.jsonl
```

Mỗi record sạch có hai field được bổ sung cho bài toán:

| Field | Ý nghĩa |
|---|---|
| `review_text` | Nội dung input, ghép từ `title` và `text` |
| `sentiment_class` | Nhãn cảm xúc tạo từ rating |

Quy tắc tạo nhãn là 1–2 sao thành Negative, 3 sao thành Neutral và 4–5 sao
thành Positive.

## 3. Kiểm chứng dữ liệu raw và kết quả làm sạch

File `data/interim/raw_data_audit.json` được tạo bằng cách đọc toàn bộ dữ liệu
raw và kiểm tra lần lượt dòng rỗng, JSON lỗi, rating không hợp lệ, review rỗng
và duplicate chính xác.

| Chỉ số kiểm chứng | Kết quả |
|---|---:|
| Tổng số dòng raw | 701.528 |
| Dòng rỗng | 0 |
| JSON lỗi/không phải object | 0 |
| Rating không hợp lệ | 0 |
| `title` và `text` cùng rỗng | 0 |
| Record trùng hoàn toàn bị loại | 7.275 |
| Record sạch được giữ lại | 694.253 |
| Đối soát tổng số dòng | Đúng (`true`) |

Phép đối soát là:

```text
701.528 = 0 + 0 + 0 + 0 + 7.275 + 694.253
```

Tất cả chênh lệch giữa raw data và cleaned data là do duplicate hoàn toàn.
SHA-256 của file raw đã kiểm tra là:

```text
36ca7c48440a8247b2dca0174d095238422b506a8a80c114dca722b7715f1c94
```

## 4. Phương pháp EDA

Script `src/eda_all_beauty.py` đọc `cleaned.jsonl` theo từng dòng để phù hợp
với dataset lớn. Với mỗi review, script:

1. Đếm rating từ 1 đến 5.
2. Đếm số record của Negative, Neutral và Positive.
3. Tính số từ trong `review_text` bằng cách tách theo khoảng trắng.
4. Tính count, mean, median, min, max, phân vị 95% cho toàn bộ review và từng
   sentiment class.
5. Tạo bốn biểu đồ và lưu các bảng kết quả.

Lệnh chạy:

```powershell
py src\eda_all_beauty.py
```

Các file kết quả EDA:

```text
outputs/metrics/eda_summary.json
outputs/metrics/eda_interpretation.md
outputs/figures/eda_rating_distribution.png
outputs/figures/eda_sentiment_distribution.png
outputs/figures/eda_review_length_distribution.png
outputs/figures/eda_review_length_by_sentiment.png
```

## 5. Phân bố rating

| Rating | Số review | Tỷ lệ trên 694.253 review sạch |
|---:|---:|---:|
| 1 sao | 100.888 | 14,53% |
| 2 sao | 42.601 | 6,14% |
| 3 sao | 55.720 | 8,03% |
| 4 sao | 78.608 | 11,32% |
| 5 sao | 416.436 | 59,98% |

Biểu đồ: `outputs/figures/eda_rating_distribution.png`.

**Diễn giải:** Rating 5 sao xuất hiện nhiều nhất, gần 60% toàn bộ review sạch.
Điều này cho thấy khách hàng trong category All_Beauty có xu hướng để lại đánh
giá tích cực. Rating 2 và 3 sao có số lượng thấp hơn rõ rệt so với rating 5,
do đó dữ liệu không phân bố đồng đều theo thang điểm.

## 6. Phân bố sentiment class

| Sentiment class | Số review | Tỷ lệ |
|---|---:|---:|
| Negative | 143.489 | 20,67% |
| Neutral | 55.720 | 8,03% |
| Positive | 495.044 | 71,31% |

Biểu đồ: `outputs/figures/eda_sentiment_distribution.png`.

**Diễn giải:** Positive là lớp chiếm đa số, với hơn 71% dữ liệu; Neutral chỉ
chiếm khoảng 8%. Đây là tình trạng class imbalance. Nếu chỉ đánh giá bằng
Accuracy, một mô hình thiên về dự đoán Positive có thể có kết quả cao nhưng
không nhận diện tốt Neutral và Negative. Vì vậy, Macro F1 được chọn làm chỉ số
chính khi so sánh mô hình ở các phase tiếp theo.

## 7. Độ dài review

Độ dài review được tính bằng số từ trong `review_text` sau khi ghép title và
text.

| Thống kê toàn bộ dữ liệu | Giá trị |
|---|---:|
| Số review | 694.253 |
| Trung bình | 36,67 từ |
| Trung vị | 23 từ |
| Nhỏ nhất | 1 từ |
| Lớn nhất | 2.594 từ |
| Phân vị 95% | 112 từ |
| Phân vị 99% | 222 từ |

Biểu đồ: `outputs/figures/eda_review_length_distribution.png`.

**Diễn giải:** Trung bình 36,67 từ cao hơn trung vị 23 từ, cho thấy phân bố độ
dài lệch phải: đa số review tương đối ngắn nhưng có một số review rất dài. Để
biểu đồ nhìn rõ phần lớn dữ liệu, histogram chỉ hiển thị review có độ dài đến
phân vị 99% (222 từ); các review dài hơn vẫn được tính trong bảng thống kê.

## 8. So sánh độ dài review theo sentiment class

| Class | Số review | Mean | Median | Min | Max | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Negative | 143.489 | 36,36 | 25 | 1 | 1.698 | 102 |
| Neutral | 55.720 | 43,58 | 30 | 1 | 2.594 | 126 |
| Positive | 495.044 | 35,98 | 22 | 1 | 1.547 | 114 |

Biểu đồ: `outputs/figures/eda_review_length_by_sentiment.png`.

**Diễn giải:** Review Neutral có độ dài trung bình 43,58 từ và trung vị 30 từ,
dài hơn cả Negative và Positive. Một giải thích hợp lý là người cho 3 sao
thường mô tả đồng thời điểm tốt và điểm chưa tốt của sản phẩm, nên cần nhiều từ
hơn. Tuy nhiên đây chỉ là quan sát từ dữ liệu; không thể kết luận rằng độ dài
là nguyên nhân tạo ra sentiment.

Boxplot ẩn outlier để hộp biểu diễn rõ khoảng giá trị của phần lớn review. Các
review rất dài vẫn được giữ trong dữ liệu và được phản ánh ở cột Max/P95.

## 9. Kết luận Phase 1

Phase 1 hoàn thành các đầu ra yêu cầu:

- Có bảng tóm tắt quá trình cleaning và bằng chứng audit.
- Có bảng phân bố rating 1–5.
- Có bảng phân bố ba sentiment classes.
- Có thống kê độ dài review toàn bộ và theo class.
- Có bốn biểu đồ EDA.
- Có diễn giải cho từng nhóm biểu đồ.

Những phát hiện cần được mang sang phase mô hình hóa là:

1. Positive chiếm đa số nên cần đánh giá bằng Macro F1, không chỉ Accuracy.
2. Neutral là lớp ít dữ liệu nhất và có thể khó dự đoán nhất.
3. Review có độ dài lệch phải; phần lớn review ngắn nhưng tồn tại một số review
   rất dài.
4. Neutral có review dài hơn trung bình, có thể chứa nội dung pha trộn.

Phase tiếp theo là phân tích chi tiết kết quả mô hình và confusion matrix, sau
đó lấy các review dự đoán sai để thực hiện error analysis.
