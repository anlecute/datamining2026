# Tóm tắt

Nghiên cứu xây dựng mô hình phân loại cảm xúc cho review sản phẩm trong category
All_Beauty của Amazon Reviews 2023. Input của bài toán là nội dung review được
ghép từ title và text; output là Negative, Neutral hoặc Positive. Nhãn được suy
ra từ rating: 1–2 sao là Negative, 3 sao là Neutral và 4–5 sao là Positive.

Từ 701.528 record raw, pipeline loại 7.275 record trùng hoàn toàn và giữ lại
694.253 review sạch. Bốn baseline được so sánh gồm Dummy Classifier,
Multinomial Naive Bayes, Logistic Regression và Linear SVM. Logistic Regression
đạt Macro F1 cao nhất là 0,7406; Linear SVM đạt Accuracy cao nhất là 0,8814.
Kết quả cũng cho thấy Neutral là lớp khó dự đoán nhất.

**Từ khóa:** sentiment analysis, Amazon Reviews, TF-IDF, text classification,
Logistic Regression, Linear SVM.

---

# Chương 1. Giới thiệu

## 1.1. Bối cảnh

Review trực tuyến là nguồn phản hồi quan trọng của khách hàng đối với sản phẩm.
Tuy nhiên, số lượng review lớn khiến việc đọc và tổng hợp thủ công trở nên khó
khả thi. Sentiment analysis giúp tự động xác định thái độ của khách hàng, hỗ
trợ nhận diện các phản hồi tích cực, tiêu cực và trung tính.

## 1.2. Mục tiêu

Project có các mục tiêu:

1. Làm sạch dữ liệu review Amazon theo quy tắc có thể kiểm chứng.
2. Tạo nhãn cảm xúc từ rating.
3. Biểu diễn text bằng TF-IDF.
4. Huấn luyện và so sánh các mô hình phân loại văn bản.
5. Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score, Macro F1 và
   confusion matrix.

## 1.3. Bài toán

Đây là bài toán supervised learning, thuộc nhóm multi-class text
classification.

| Thành phần | Nội dung |
|---|---|
| Input | `review_text = title + " " + text` |
| Output | Negative / Neutral / Positive |
| Feature | Vector TF-IDF tạo từ `review_text` |
| Label | `sentiment_class` |

| Rating | Nhãn sentiment |
|---:|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

Rating chỉ dùng để sinh label, không đưa vào feature. Nếu dùng rating khi train,
model sẽ biết trước đáp án và gây data leakage.

---

# Chương 2. Dữ liệu

## 2.1. Nguồn và lựa chọn dữ liệu

Dataset sử dụng là Amazon Reviews 2023 do McAuley Lab công bố. Nghiên cứu chọn
category **All_Beauty** và chỉ dùng file **User Reviews**.

Nguồn: <https://amazon-reviews-2023.github.io/>

File raw được lưu tại `C:\AnLe\datamining\All_Beauty.jsonl\All_Beauty.jsonl`.
User Reviews được chọn vì có đủ ba trường cần thiết là `rating`, `title` và
`text`. Item Metadata, rating-only và timestamp split không phù hợp với
baseline text classification vì không có nội dung review đầy đủ.

## 2.2. Thuộc tính sử dụng

| Attribute | Vai trò trong project |
|---|---|
| `rating` | Sinh nhãn `sentiment_class` |
| `title` | Ghép vào `review_text` |
| `text` | Ghép vào `review_text` |
| `user_id`, `asin`, `parent_asin`, `timestamp` | Được giữ lại, chưa dùng làm feature |
| `verified_purchase`, `helpful_vote` | Không dùng trong text-only baseline |

## 2.3. Kiểm chứng file raw

File `data/interim/raw_data_audit.json` kiểm chứng toàn bộ dữ liệu raw theo các
quy tắc tiền xử lý.

| Chỉ số | Kết quả |
|---|---:|
| Tổng số dòng raw | 701.528 |
| Dòng rỗng | 0 |
| JSON lỗi/không phải object | 0 |
| Rating không hợp lệ | 0 |
| Title và text cùng rỗng | 0 |
| Duplicate hoàn toàn | 7.275 |
| Record được giữ lại | 694.253 |
| Đối soát tổng số dòng | Đúng |

SHA-256 của file raw là `36ca7c48440a8247b2dca0174d095238422b506a8a80c114dca722b7715f1c94`.

---

# Chương 3. Tiền xử lý dữ liệu

## 3.1. Các quy tắc tiền xử lý

Mỗi dòng JSONL được xử lý theo thứ tự sau:

1. Bỏ dòng trống hoặc JSON không hợp lệ.
2. Chỉ nhận rating thuộc 1, 2, 3, 4 hoặc 5.
3. Giữ review nếu title hoặc text có nội dung; bỏ nếu cả hai cùng rỗng.
4. Loại record chỉ khi toàn bộ JSON record trùng hoàn toàn.
5. Chuẩn hóa khoảng trắng, ghép title/text thành `review_text`.
6. Sinh `sentiment_class` từ rating.

Record sạch vẫn giữ toàn bộ field gốc và có thêm `review_text` cùng
`sentiment_class`.

## 3.2. Kết quả tiền xử lý

| Chỉ số | Số lượng |
|---|---:|
| Record raw | 701.528 |
| Duplicate bị loại | 7.275 |
| Record sạch | 694.253 |

Các artifact đầu ra là `cleaned.jsonl`, `preprocessing_stats.json`,
`amazon_sentiment.arff` và `amazon_sentiment.csv`.

---

# Chương 4. Khảo sát dữ liệu (EDA)

> Thứ tự trình bày output EDA là: cleaning summary → phân bố rating → phân bố
> sentiment → độ dài review → độ dài review theo sentiment.

## 4.1. Phân bố rating

**Hình 4.1. Phân bố rating của review**  
Chèn: `outputs/figures/eda_rating_distribution.png`

| Rating | Số review | Tỷ lệ |
|---:|---:|---:|
| 1 sao | 100.888 | 14,53% |
| 2 sao | 42.601 | 6,14% |
| 3 sao | 55.720 | 8,03% |
| 4 sao | 78.608 | 11,32% |
| 5 sao | 416.436 | 59,98% |

Rating 5 sao chiếm gần 60% toàn bộ review sạch, cao hơn đáng kể các mức rating
còn lại. Điều này dự báo lớp Positive sẽ chiếm đa số sau khi tạo nhãn.

## 4.2. Phân bố sentiment class

**Hình 4.2. Phân bố ba lớp cảm xúc**  
Chèn: `outputs/figures/eda_sentiment_distribution.png`

| Class | Số review | Tỷ lệ |
|---|---:|---:|
| Negative | 143.489 | 20,67% |
| Neutral | 55.720 | 8,03% |
| Positive | 495.044 | 71,31% |

Positive chiếm hơn 71% dữ liệu, còn Neutral chỉ chiếm khoảng 8%. Đây là class
imbalance; Macro F1 được chọn làm chỉ số chính thay vì chỉ dùng Accuracy.

## 4.3. Độ dài review toàn bộ

**Hình 4.3. Phân bố độ dài review**  
Chèn: `outputs/figures/eda_review_length_distribution.png`

| Thống kê | Giá trị |
|---|---:|
| Mean | 36,67 từ |
| Median | 23 từ |
| Min | 1 từ |
| Max | 2.594 từ |
| P95 | 112 từ |
| P99 | 222 từ |

Mean lớn hơn median cho thấy phân bố lệch phải: phần lớn review ngắn, nhưng có
một số review rất dài. Histogram hiển thị đến P99 để các outlier dài không làm
che phân bố chính.

## 4.4. Độ dài review theo sentiment

**Hình 4.4. Boxplot độ dài review theo sentiment class**  
Chèn: `outputs/figures/eda_review_length_by_sentiment.png`

| Class | Mean | Median | P95 |
|---|---:|---:|---:|
| Negative | 36,36 | 25 | 102 |
| Neutral | 43,58 | 30 | 126 |
| Positive | 35,98 | 22 | 114 |

Review Neutral dài hơn Negative và Positive theo cả mean lẫn median. Một giải
thích hợp lý là người cho 3 sao thường mô tả đồng thời ưu điểm và nhược điểm.
Đây là quan sát từ dữ liệu, không khẳng định quan hệ nhân quả.

---

# Chương 5. Phương pháp mô hình hóa

## 5.1. Chia dữ liệu

Từ 694.253 review sạch, 50.000 review được lấy mẫu ngẫu nhiên bằng uniform
reservoir sampling với `random_state=42`. Reservoir sampling đọc file JSONL một
lần nhưng vẫn bảo đảm mỗi review có cơ hội như nhau được chọn; vì vậy không chỉ
lấy 50.000 dòng đầu file. Mẫu sau đó được chia stratified theo tỷ lệ 80/20.

| Tập | Tỷ lệ | Số review |
|---|---:|---:|
| Train | 80% | 40.000 |
| Test | 20% | 10.000 |

Stratified split giữ tỷ lệ ba lớp gần giống dữ liệu tổng thể. Số lượng class
thực tế trong thí nghiệm là:

| Class | Mẫu 50.000 | Train | Test |
|---|---:|---:|---:|
| Negative | 10.348 | 8.278 | 2.070 |
| Neutral | 4.089 | 3.271 | 818 |
| Positive | 35.563 | 28.451 | 7.112 |
| Tổng | 50.000 | 40.000 | 10.000 |

`random_state=42` khóa phần ngẫu nhiên của sampling và split. Nhờ đó, nếu dữ
liệu và code không đổi, lần chạy sau dùng cùng sample và cùng train/test split.
Toàn bộ cấu hình được lưu tại `outputs/metrics/experiment_configuration.json`.

## 5.2. TF-IDF

TF-IDF chỉ được fit trên train set; test set chỉ được transform để tránh data
leakage.

| Tham số | Giá trị |
|---|---|
| `lowercase` | `True` |
| `ngram_range` | `(1, 2)` |
| `min_df` | `2` |
| `max_df` | `0.95` |
| `max_features` | `20.000` |

Unigram và bigram giúp nhận diện các cụm quan trọng như `not good`, `very bad`
và `highly recommend`.

Trình tự đúng là `fit_transform(x_train)` rồi `transform(x_test)`. Nếu
`fit_transform` TF-IDF trên toàn bộ review trước khi split, vocabulary và trọng
số IDF đã nhận thông tin từ test set; đây là data leakage và kết quả đánh giá sẽ
lạc quan hơn chất lượng thực tế.

## 5.3. Các mô hình và chỉ số

| Mô hình | Cấu hình chính |
|---|---|
| Dummy Classifier | `most_frequent` |
| Multinomial Naive Bayes | Mặc định |
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Linear SVM | `class_weight="balanced"` |

Các chỉ số gồm Accuracy, Macro F1 và Weighted F1. Macro F1 là tiêu chí chọn
model vì nó đánh giá ngang bằng ba lớp.

## 5.4. Tránh target leakage từ rating

Rating được dùng để tạo `sentiment_class`, nên rating không được xuất hiện trong
feature đầu vào. Nếu input chứa `rating=5`, model biết ngay label là Positive
theo quy tắc gán nhãn, dù không cần hiểu nội dung review. Vì vậy feature duy
nhất của baseline là `review_text`, còn rating chỉ phục vụ bước tạo label.

## 5.5. Mất cân bằng lớp và chọn chỉ số

Positive chiếm 71,31% dữ liệu sạch trong khi Neutral chỉ chiếm 8,03%. Một model
luôn đoán Positive có thể có Accuracy khoảng 71% nhưng không có giá trị trong
việc nhận diện Neutral hoặc Negative. Do đó Macro F1 được dùng làm chỉ số chọn
model. Logistic Regression và Linear SVM dùng `class_weight="balanced"` để tăng
mức phạt khi dự đoán sai các lớp ít mẫu.

---

# Chương 6. Kết quả thực nghiệm

> Thứ tự trình bày output mô hình là: model comparison → classification report
> của model được chọn → confusion matrix → thảo luận.

## 6.1. So sánh các mô hình

Nguồn: `outputs/metrics/model_comparison.csv`.

| Mô hình | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dummy (most frequent) | 0,7112 | 0,2771 | 0,5912 |
| Multinomial Naive Bayes | 0,8665 | 0,6266 | 0,8367 |
| Logistic Regression | 0,8632 | **0,7406** | 0,8733 |
| Linear SVM | **0,8814** | 0,7353 | **0,8803** |

Dummy có Accuracy 0,7112 do luôn đoán lớp Positive, nhưng Macro F1 rất thấp.
Điều này xác nhận Accuracy đơn lẻ không phù hợp với dữ liệu mất cân bằng.

Logistic Regression có Macro F1 cao nhất nên được chọn làm model chính. Linear
SVM có Accuracy cao nhất nhưng Macro F1 thấp hơn 0,0053. Vì Macro F1 coi ba
class quan trọng như nhau, Logistic Regression cân bằng tốt hơn cho mục tiêu
phân loại Negative, Neutral và Positive; không chỉ ưu tiên Positive là class đa
số.

## 6.2. Báo cáo theo class của Logistic Regression

Nguồn: `outputs/metrics/best_model_report.txt`.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0,81 | 0,83 | 0,82 | 2.070 |
| Neutral | 0,39 | 0,59 | 0,47 | 818 |
| Positive | 0,97 | 0,91 | 0,94 | 7.112 |
| Macro average | 0,72 | 0,77 | 0,74 | 10.000 |
| Weighted average | 0,89 | 0,86 | 0,87 | 10.000 |

Positive là lớp được nhận diện tốt nhất. Neutral có F1 thấp nhất, phù hợp với
EDA: đây là lớp ít dữ liệu nhất và review của lớp này thường dài, pha trộn cảm
xúc.

## 6.3. Confusion matrix

**Hình 6.1. Confusion matrix của Logistic Regression**  
Chèn: `outputs/figures/confusion_matrix_best_model.png`

Confusion matrix cần được dùng để mô tả các cặp class mà model nhầm lẫn. Do
Neutral có F1 thấp nhất, error analysis nên tập trung vào các review Neutral bị
nhầm sang Positive hoặc Negative, và các review lớp khác bị đoán là Neutral.

---

# Chương 7. Thảo luận, hạn chế và hướng phát triển

## 7.1. Nhận xét chính

1. Dữ liệu mất cân bằng mạnh về Positive; Macro F1 phù hợp hơn Accuracy.
2. Mô hình tuyến tính với TF-IDF tốt hơn rõ rệt Dummy baseline.
3. Logistic Regression cân bằng hiệu năng ba lớp tốt nhất.
4. Neutral là lớp khó nhất với F1-score 0,47.

## 7.2. Hạn chế

- Nhãn được suy ra từ rating, không phải sentiment label do con người gán tay.
- Review 3 sao có thể chứa cả điểm tốt và điểm chưa tốt, khiến Neutral khó xác
  định.
- Kết quả hiện tại dùng mẫu 50.000 record; chưa thử toàn bộ 694.253 record.
- Chưa thực hiện error analysis định tính trên các review dự đoán sai.

## 7.3. Hướng phát triển

1. Phân tích 20–30 review dự đoán sai.
2. So sánh unigram với unigram + bigram.
3. Tinh chỉnh tham số `C` cho Logistic Regression và Linear SVM.
4. Train lại trên toàn bộ cleaned dataset sau khi chốt cấu hình.
5. Sau baseline, có thể nghiên cứu embedding hoặc BERT.

---

# Chương 8. Kết luận

Project đã xây dựng pipeline tái lập từ raw JSONL đến clean data, ARFF, EDA,
huấn luyện và đánh giá. Từ 701.528 record raw, pipeline giữ lại 694.253 record
sau khi loại 7.275 duplicate hoàn toàn.

EDA cho thấy Positive chiếm 71,31% còn Neutral chỉ chiếm 8,03%. Vì vậy, Macro
F1 được chọn làm chỉ số chính. Trong bốn baseline, Logistic Regression đạt Macro
F1 tốt nhất là 0,7406 và được chọn làm model chính. Neutral là class có hiệu
năng thấp nhất và là trọng tâm cần phân tích ở bước tiếp theo.

---

# Phụ lục. Thứ tự dùng các output trong report

| Artifact | Vị trí trong report | Cách sử dụng |
|---|---|---|
| `raw_data_audit.json` | Chương 2–3 | Bằng chứng file raw và cleaning |
| `preprocessing_stats.json` | Chương 3 | Bảng kết quả tiền xử lý |
| `eda_summary.json` | Chương 4 | Nguồn số liệu EDA |
| `eda_rating_distribution.png` | Chương 4.1 | Hình 4.1 |
| `eda_sentiment_distribution.png` | Chương 4.2 | Hình 4.2 |
| `eda_review_length_distribution.png` | Chương 4.3 | Hình 4.3 |
| `eda_review_length_by_sentiment.png` | Chương 4.4 | Hình 4.4 |
| `experiment_configuration.json` | Chương 5 | Bằng chứng cấu hình tái lập |
| `model_comparison.csv` | Chương 6.1 | Bảng so sánh model |
| `best_model_report.txt` | Chương 6.2 | Metrics theo class |
| `confusion_matrix_best_model.png` | Chương 6.3 | Hình 6.1 |
