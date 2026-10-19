# Report Phase 2 — Xác nhận phương pháp mô hình hóa

## Project: Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu Phase 2

Phase 2 xác nhận mô hình học từ dữ liệu nào, quy trình train/test được thực hiện
ra sao và vì sao kết quả có thể tin cậy. Mục tiêu không phải tạo thêm model mới,
mà là mô tả đầy đủ phương pháp đã dùng để người khác có thể chạy lại thí nghiệm
và nhận được kết quả tương tự.

Các câu hỏi cần trả lời là:

1. Feature và label của bài toán là gì?
2. Dữ liệu được lấy mẫu, chia train/test như thế nào?
3. TF-IDF biến text thành dữ liệu số ra sao?
4. Làm thế nào để tránh data leakage và overfitting?
5. Vì sao cần Dummy Classifier, nhiều model và Macro F1?
6. Vì sao không dùng rating làm feature?
7. Vì sao Logistic Regression được chọn thay vì Linear SVM?

## 2. Bằng chứng cấu hình thí nghiệm

File `outputs/metrics/experiment_configuration.json` được tạo lại khi chạy
`py src\train_models.py`. File này lưu toàn bộ cấu hình dùng trong thí nghiệm,
bao gồm nguồn input, random seed, phương pháp lấy mẫu, split, TF-IDF, models và
tiêu chí chọn model.

| Thành phần | Cấu hình thực tế |
|---|---|
| File đầu vào | `data/interim/cleaned.jsonl` |
| Tổng review sạch | 694.253 |
| Số review dùng train/test | 50.000 |
| Cách lấy mẫu | Uniform reservoir sampling |
| Random seed | 42 |
| Train/test split | Stratified 80/20 |
| Train set | 40.000 review |
| Test set | 10.000 review |
| Feature extraction | TF-IDF unigram + bigram |
| Tiêu chí chọn model | Macro F1 |

Lệnh tái lập toàn bộ Phase 2:

```powershell
py src\train_models.py
```

Lệnh mặc định luôn dùng `random_state=42`. Nếu code, dữ liệu đầu vào và phiên
bản thư viện không đổi, lần chạy lại dùng cùng mẫu 50.000 review và cùng cách
chia train/test.

## 3. Bài toán học có giám sát

Đây là bài toán **supervised learning** vì mỗi review trong dữ liệu train đã có
nhãn. Đây cũng là bài toán **multi-class classification** vì model phải chọn một
trong ba lớp.

| Khái niệm | Áp dụng trong project |
|---|---|
| Instance/sample | Một review Amazon |
| Feature/input `X` | `review_text` sau khi chuyển sang vector TF-IDF |
| Label/target `y` | `sentiment_class` |
| Classes | Negative, Neutral, Positive |
| Train set | Dùng để fit TF-IDF và train model |
| Test set | Chỉ dùng một lần để đánh giá model đã train |

Một record có thể có nhiều field gốc, nhưng model baseline chỉ học từ text. Ví
dụ, `user_id`, `asin`, timestamp và helpful_vote không được đưa vào feature.

## 4. Vì sao không dùng rating làm feature

Rating được dùng để tạo label theo quy tắc 1–2 → Negative, 3 → Neutral, 4–5 →
Positive. Nếu rating xuất hiện trong `X`, model có thể dự đoán nhãn chỉ bằng
rating mà không cần hiểu review text.

Ví dụ, khi input chứa `rating=5`, label luôn là Positive theo quy tắc tạo nhãn.
Model có thể đạt kết quả gần như hoàn hảo nhưng không hề học được nghĩa của các
câu như “The product broke after one day”. Đây là **target leakage**: feature
chứa trực tiếp thông tin dùng để tạo target.

Vì vậy pipeline dùng:

```text
X = review_text
y = sentiment_class được tạo từ rating
```

Chứ không dùng:

```text
X = rating + review_text   ← sai vì leakage
y = sentiment_class
```

## 5. Lấy mẫu và chia train/test

### 5.1. Reservoir sampling

Dataset sạch có 694.253 review. Để baseline chạy nhanh mà vẫn đại diện cho toàn
bộ dữ liệu, script lấy ngẫu nhiên 50.000 review bằng **uniform reservoir
sampling**. Kỹ thuật này đọc file JSONL một lần, không cần nạp toàn bộ text vào
RAM, và mỗi record có cơ hội như nhau được chọn vào mẫu cuối cùng.

### 5.2. Stratified train/test split

50.000 review được chia 80% train và 20% test bằng:

```python
train_test_split(..., test_size=0.2, random_state=42, stratify=labels)
```

Số lượng từng class trước và sau khi chia:

| Class | Mẫu 50.000 | Train | Test |
|---|---:|---:|---:|
| Negative | 10.348 | 8.278 | 2.070 |
| Neutral | 4.089 | 3.271 | 818 |
| Positive | 35.563 | 28.451 | 7.112 |
| Tổng | 50.000 | 40.000 | 10.000 |

`stratify=labels` rất quan trọng vì Neutral là lớp ít dữ liệu. Nếu chia ngẫu
nhiên không stratify, test set có thể có quá ít Neutral, khiến F1-score của lớp
này không ổn định hoặc không phản ánh đúng model.

### 5.3. Vai trò của random_state=42

Random sampling và train/test split có yếu tố ngẫu nhiên. `random_state=42`
khóa trạng thái ngẫu nhiên, nhờ đó lần chạy sau sẽ có cùng sample và cùng split.
Điều này giúp so sánh công bằng giữa các model hoặc giữa các cấu hình TF-IDF.
Số 42 không có ý nghĩa đặc biệt; bất kỳ số nguyên cố định nào cũng được, miễn
được ghi lại trong report.

## 6. Biến review text thành feature bằng TF-IDF

Machine Learning model không hiểu trực tiếp chuỗi ký tự. `TfidfVectorizer`
chuyển mỗi review thành một vector số, trong đó một term quan trọng với review
và hiếm trong toàn bộ train set sẽ có trọng số cao hơn.

Trực giác của TF-IDF:

```text
TF-IDF(term, document) = số lần term xuất hiện trong document
                          × độ hiếm của term trong các document train
```

Ví dụ, các term như `product` có thể xuất hiện trong nhiều review nên ít phân
biệt class. Các term như `refund`, `broken`, `excellent` hoặc `highly
recommend` có thể mang nhiều tín hiệu sentiment hơn.

### 6.1. Cấu hình TF-IDF

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `lowercase` | `True` | Xem “Great” và “great” là cùng một term |
| `ngram_range` | `(1, 2)` | Dùng unigram và bigram |
| `min_df` | `2` | Loại term chỉ xuất hiện một lần trong train set |
| `max_df` | `0.95` | Loại term xuất hiện trong hơn 95% review train |
| `max_features` | `20.000` | Giới hạn vocabulary để kiểm soát tài nguyên |

**Unigram** là một từ, ví dụ `great`. **Bigram** là hai từ liên tiếp, ví dụ
`not good`. Bigram quan trọng trong sentiment analysis vì từ `good` riêng lẻ là
tích cực, nhưng cụm `not good` lại mang nghĩa tiêu cực.

## 7. Chống data leakage

Data leakage là khi thông tin của test set ảnh hưởng vào quá trình train. Khi
đó kết quả đánh giá sẽ lạc quan hơn chất lượng thực tế của model.

Pipeline hiện tại làm đúng theo thứ tự:

```text
1. Chia review text thành train và test.
2. Fit TF-IDF chỉ trên x_train.
3. Transform x_train bằng vectorizer đã fit.
4. Transform x_test bằng chính vectorizer đó, không fit lại.
5. Train model bằng feature của train set.
6. Đánh giá một lần trên test set.
```

Mã logic tương ứng là:

```python
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
```

Cách làm sai là `fit_transform` trên toàn bộ review trước khi split. Khi đó
IDF và vocabulary đã “nhìn thấy” test set. Pipeline hiện tại không mắc lỗi này.

## 8. Baseline và các model được so sánh

| Model | Vai trò | Cấu hình |
|---|---|---|
| Dummy Classifier | Mốc tối thiểu; luôn đoán class phổ biến nhất | `most_frequent` |
| Multinomial Naive Bayes | Baseline phổ biến cho text sparse | Mặc định |
| Logistic Regression | Model tuyến tính mạnh, ổn định | `max_iter=1000`, `class_weight="balanced"` |
| Linear SVM | Model tuyến tính hiệu quả với TF-IDF sparse | `class_weight="balanced"` |

Dummy Classifier là cần thiết. Nếu Dummy đã có Accuracy cao vì lớp Positive
chiếm đa số, model thật phải vượt Dummy một cách đáng kể mới có giá trị.

## 9. Mất cân bằng lớp và metrics

Phân bố toàn bộ dữ liệu sạch là Positive 71,31%, Negative 20,67% và Neutral
8,03%. Nếu luôn dự đoán Positive, model có thể đạt Accuracy gần 71% nhưng bỏ
qua hai lớp còn lại.

| Metric | Ý nghĩa | Vai trò trong project |
|---|---|---|
| Accuracy | Tỷ lệ dự đoán đúng trên toàn bộ test set | Báo cáo bổ sung |
| Precision | Dự đoán một class đúng bao nhiêu | Phân tích từng class |
| Recall | Nhận ra được bao nhiêu record thật của class | Phân tích từng class |
| F1-score | Cân bằng Precision và Recall | Phân tích từng class |
| Macro F1 | Trung bình F1, ba class có trọng số bằng nhau | Tiêu chí chọn model |
| Weighted F1 | F1 có trọng số theo support của class | Báo cáo bổ sung |

`class_weight="balanced"` trong Logistic Regression và Linear SVM tăng mức
phạt khi dự đoán sai class ít mẫu. Đây là một cách xử lý imbalance trên train
set mà không sao chép hoặc xóa review.

## 10. Lý do chọn Logistic Regression

Kết quả train lại với cấu hình tái lập:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Dummy (most frequent) | 0,7112 | 0,2771 | 0,5912 |
| Multinomial Naive Bayes | 0,8665 | 0,6266 | 0,8367 |
| Logistic Regression | 0,8632 | **0,7406** | 0,8733 |
| Linear SVM | **0,8814** | 0,7353 | **0,8803** |

Linear SVM có Accuracy cao hơn 0,0182, nhưng Logistic Regression có Macro F1
cao hơn 0,0053. Vì Macro F1 là trung bình F1 của Negative, Neutral và Positive
với trọng số ngang nhau, nó phản ánh tốt hơn chất lượng trên Neutral. Do đó,
Logistic Regression được chọn làm model chính.

Quyết định này không có nghĩa Linear SVM là model kém; Linear SVM là model có
Accuracy cao nhất. Tuy nhiên mục tiêu project là phân loại tốt cả ba class,
không chỉ tối ưu lớp Positive đa số.

## 11. Overfitting và độ tin cậy của kết quả

Overfitting xảy ra khi model học quá sát dữ liệu train nhưng hoạt động kém với
dữ liệu mới. Pipeline giảm rủi ro này bằng cách đánh giá trên 10.000 review test
không tham gia vào fit TF-IDF hoặc train model.

Kết quả hiện tại là baseline trên một split cố định. Để đánh giá chắc chắn hơn
ở giai đoạn mở rộng, có thể thực hiện cross-validation trên train set hoặc thử
nhiều random seed. Tuy nhiên, test set phải được giữ riêng đến bước đánh giá
cuối cùng để không biến nó thành validation set.

## 12. Mã giả quy trình mô hình hóa

```text
INPUT: cleaned.jsonl gồm review_text và sentiment_class

đọc từng record trong cleaned.jsonl
lấy uniform reservoir sample gồm 50.000 record với seed 42
chia sample thành train 80% và test 20%, stratify theo label

fit TF-IDF (unigram + bigram) chỉ trên review_text của train
transform review_text của train và test bằng TF-IDF đã fit

khởi tạo Dummy, Naive Bayes, Logistic Regression, Linear SVM
với mỗi model:
    train trên feature train
    dự đoán test set
    tính Accuracy, Macro F1 và Weighted F1

chọn model có Macro F1 cao nhất
lưu cấu hình thí nghiệm, bảng metrics, classification report và confusion matrix
```

## 13. Kết luận Phase 2

Phase 2 đã xác nhận đầy đủ phương pháp mô hình hóa:

- Feature là `review_text`; label là `sentiment_class`.
- Rating không được dùng làm feature vì sẽ gây target leakage.
- Dữ liệu được lấy mẫu ngẫu nhiên, chia stratified 80/20 và cố định seed 42.
- TF-IDF chỉ fit trên train set, nên test set không làm rò rỉ vocabulary hoặc
  IDF.
- Dummy Classifier là baseline tối thiểu; ba model còn lại được so sánh công
  bằng trên cùng dữ liệu test.
- Macro F1 và `class_weight="balanced"` được dùng để xử lý tình trạng mất cân
  bằng lớp.
- Logistic Regression được chọn vì Macro F1 cao nhất, phù hợp với mục tiêu
  đánh giá cân bằng cả Negative, Neutral và Positive.

Artifact chứng minh cấu hình Phase 2 là
`outputs/metrics/experiment_configuration.json`. Bước tiếp theo là Phase 3:
phân tích bảng kết quả, classification report và confusion matrix chi tiết.
