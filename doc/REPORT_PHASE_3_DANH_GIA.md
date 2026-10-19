# Report Phase 3 — Phân tích kết quả baseline và lỗi mô hình

## Project: Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu Phase 3

Phase 3 chuyển các metric thành nhận xét có ý nghĩa. Việc đánh giá được thực
hiện trên 10.000 review thuộc test set, tức là các review không tham gia train
model hoặc fit TF-IDF. Vì các review này có `sentiment_class` thật, dự đoán của
model có thể được so sánh trực tiếp với nhãn thật.

Các artifact dùng để đánh giá:

| File | Nội dung |
|---|---|
| `outputs/metrics/model_comparison.csv` | Accuracy, Macro F1, Weighted F1 của 4 model |
| `outputs/metrics/best_model_report.txt` | Precision, Recall, F1-score theo class |
| `outputs/figures/confusion_matrix_best_model.png` | Confusion matrix model tốt nhất |
| `outputs/metrics/error_analysis_report.md` | Tổng hợp các cặp nhầm lẫn |
| `outputs/metrics/error_examples.csv` | 30 review dự đoán sai để đọc định tính |

Lệnh tái tạo kết quả:

```powershell
py src\train_models.py
py src\analyze_errors.py
```

## 2. So sánh bốn baseline

| Mô hình | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dummy Classifier | 0,7112 | 0,2771 | 0,5912 |
| Multinomial Naive Bayes | 0,8665 | 0,6266 | 0,8367 |
| Logistic Regression | 0,8632 | **0,7406** | 0,8733 |
| Linear SVM | **0,8814** | 0,7353 | **0,8803** |

### Nhận xét 1 — Baseline Dummy cho thấy Accuracy có thể gây hiểu lầm

Dummy Classifier luôn dự đoán class phổ biến nhất là Positive. Nó vẫn đạt
Accuracy 0,7112 vì Positive chiếm 71,31% dữ liệu, nhưng Macro F1 chỉ đạt 0,2771.
Điều này cho thấy Accuracy đơn lẻ không phản ánh khả năng nhận diện đầy đủ ba
class trong dữ liệu mất cân bằng.

### Nhận xét 2 — Các model học từ text tốt hơn đáng kể Dummy baseline

Multinomial Naive Bayes, Logistic Regression và Linear SVM đều có Macro F1 cao
hơn Dummy. Logistic Regression tăng Macro F1 từ 0,2771 lên 0,7406, tương đương
tăng 0,4635. Kết quả này là bằng chứng rằng model học được tín hiệu sentiment
từ `review_text`, thay vì chỉ dự đoán class phổ biến.

### Nhận xét 3 — Logistic Regression được chọn theo Macro F1

Linear SVM có Accuracy cao nhất là 0,8814, nhưng Logistic Regression có Macro
F1 cao hơn: 0,7406 so với 0,7353. Vì Macro F1 coi Negative, Neutral và Positive
có trọng số ngang nhau, Logistic Regression được chọn làm model chính. Lựa chọn
này ưu tiên khả năng phân loại cân bằng ba class, không chỉ tối ưu lớp Positive
là class đa số.

## 3. Đánh giá theo từng class

Classification report của Logistic Regression:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0,81 | 0,83 | 0,82 | 2.070 |
| Neutral | 0,39 | 0,59 | 0,47 | 818 |
| Positive | 0,97 | 0,91 | 0,94 | 7.112 |
| Macro average | 0,72 | 0,77 | 0,74 | 10.000 |
| Weighted average | 0,89 | 0,86 | 0,87 | 10.000 |

### Nhận xét 4 — Neutral là class khó nhất

Neutral có F1-score 0,47, thấp hơn Negative (0,82) và Positive (0,94). Precision
của Neutral chỉ là 0,39, nghĩa là trong các review model dự đoán Neutral, một
phần đáng kể thực sự thuộc class khác. Lớp này cũng ít dữ liệu nhất trong test
set, chỉ có 818 review.

### Nhận xét 5 — Positive được nhận diện tốt nhất

Positive có F1-score 0,94, Precision 0,97 và Recall 0,91. Kết quả cao một phần
do Positive có nhiều dữ liệu train nhất, đồng thời các review Positive thường có
từ/cụm từ biểu đạt cảm xúc rõ hơn.

## 4. Confusion matrix và các cặp nhầm lẫn

**Hình cần chèn:** `outputs/figures/confusion_matrix_best_model.png`.

Confusion matrix dưới đây có hàng là nhãn thật và cột là nhãn model dự đoán.

| Nhãn thật \ Dự đoán | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1.708 | 278 | 84 |
| Neutral | 209 | 481 | 128 |
| Positive | 188 | 481 | 6.443 |

Tổng số dự đoán đúng là 8.632 trên 10.000 review, tương đương Accuracy 0,8632.
Có 1.368 dự đoán sai, tương đương error rate 13,68%.

| Nhãn thật | Bị dự đoán thành | Số lỗi |
|---|---|---:|
| Positive | Neutral | 481 |
| Negative | Neutral | 278 |
| Neutral | Negative | 209 |
| Positive | Negative | 188 |
| Neutral | Positive | 128 |
| Negative | Positive | 84 |

### Nhận xét 6 — Model thường dự đoán nhầm sang Neutral

Các lỗi Positive → Neutral và Negative → Neutral lần lượt là 481 và 278 review.
Điều này cho thấy khi review chứa cả tín hiệu tốt và chưa tốt, hoặc diễn đạt
không mạnh, model có xu hướng dự đoán Neutral.

### Nhận xét 7 — Neutral bị nhầm nhiều hơn các class khác

Trong 818 review Neutral, chỉ 481 review được dự đoán đúng, tương đương Recall
58,80%. Có 209 review Neutral bị đoán thành Negative và 128 review bị đoán
thành Positive. Điều này phù hợp với bản chất nhãn 3 sao: nội dung có thể nghiêng
về khen hoặc chê, dù rating vẫn là 3 sao.

## 5. Phân tích định tính các ví dụ lỗi

File `error_examples.csv` chứa tối đa năm ví dụ cho mỗi cặp nhầm lẫn; các ví dụ
được chọn ngẫu nhiên với seed 42 để có thể tái lập. Đây là ví dụ minh họa định
tính, không phải thống kê toàn bộ test set.

### 5.1. Negative bị dự đoán thành Neutral

Một số review có lời khen nhỏ đi kèm phàn nàn chính, ví dụ sản phẩm “build
quality was ok” nhưng “not wood at all”, hoặc “liked the size” nhưng rhinestones
“fall out easily”. Sự hiện diện đồng thời của từ tích cực và tiêu cực khiến
model có thể thiên về Neutral, dù rating thuộc Negative.

### 5.2. Neutral bị dự đoán thành Negative hoặc Positive

Review 3 sao có thể chứa ngôn ngữ khá tiêu cực như “doesn't look like photo”
hoặc “Not for me”, nên model đoán Negative. Ngược lại, một review nói “still
very pleased with the purchase” hoặc “Awesome exfoliation” có thể bị đoán
Positive dù rating thực tế là 3 sao. Đây là giới hạn của việc suy nhãn sentiment
từ rating: rating và cảm xúc trong từng câu không phải lúc nào cũng hoàn toàn
trùng khớp.

### 5.3. Positive bị dự đoán thành Neutral hoặc Negative

Một số review Positive có góp ý hoặc phàn nàn cụ thể, chẳng hạn sản phẩm “does
work” nhưng “neither intuitive nor easy to use”, hoặc “smells really good” nhưng
không đúng loại texture mong đợi. Những từ phủ định hoặc mô tả vấn đề có thể
làm model giảm mức tự tin và chuyển dự đoán sang Neutral/Negative.

## 6. Kết luận Phase 3

Phase 3 hoàn thành các yêu cầu đánh giá baseline:

- Có bảng so sánh bốn model bằng Accuracy, Macro F1 và Weighted F1.
- Có classification report theo từng class.
- Có confusion matrix và bảng số lỗi cho từng cặp class.
- Có nhiều hơn ba nhận xét được chứng minh trực tiếp bằng số liệu.
- Có các ví dụ lỗi định tính từ test set.

Kết luận chính là Logistic Regression phù hợp nhất theo Macro F1, nhưng Neutral
vẫn là class yếu nhất. Bước tiếp theo nên là Phase 4: phân tích lỗi sâu hơn và
thử một số cấu hình TF-IDF/model có kiểm soát để xem Macro F1 của Neutral có
cải thiện hay không.
