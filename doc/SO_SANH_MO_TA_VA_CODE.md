# Đối chiếu yêu cầu ban đầu và implementation hiện tại

## 1. Mục đích và phạm vi

Tài liệu này đối chiếu yêu cầu ban đầu của project với source code và các
artifact hiện có của project Amazon Reviews 2023, category `All_Beauty`.

Mục tiêu là phân biệt ba loại nội dung:

1. **Yêu cầu đã đáp ứng** — code và artifact thực hiện đúng yêu cầu ban đầu.
2. **Lựa chọn thiết kế cần nêu rõ** — không sai, nhưng cần được giải thích trong
   report để người đọc hiểu phạm vi thí nghiệm.
3. **Điểm cần đồng bộ hoặc cải thiện** — chủ yếu là tính nhất quán của report
   hoặc các thử nghiệm mở rộng.

Nguồn đã đối chiếu:

- `src/preprocess_all_beauty.py`
- `src/audit_raw_data.py`
- `src/export_arff.py`
- `src/eda_all_beauty.py`
- `src/train_models.py`
- `src/tune_models.py`
- các artifact trong `data/interim/` và `outputs/metrics/`

---

## 2. Kết luận ngắn

Pipeline cốt lõi đã đáp ứng yêu cầu ban đầu: chọn đúng dữ liệu User Reviews,
làm sạch dữ liệu, tạo ba nhãn cảm xúc, tạo `review_text`, xuất `cleaned.jsonl`
và ARFF, khảo sát EDA, đồng thời huấn luyện các mô hình phân loại văn bản.

Không có bằng chứng cho thấy pipeline tiền xử lý hoặc mô hình hóa bị sai về mặt
logic. Các việc cần ưu tiên là:

1. Thống nhất **mô hình cuối** trong tất cả report là Linear SVM với `C=0.5`.
2. Nêu rõ **duplicate** nghĩa là hai JSON record giống hệt nhau.
3. Nêu rõ kết quả baseline/tuning dùng mẫu ngẫu nhiên 50.000 review, không phải
   toàn bộ 694.253 review sạch.
4. Nếu phần Naive Bayes trong bài nộp cần bám sát công thức đếm từ đã trình bày,
   bổ sung một thí nghiệm `CountVectorizer` + `MultinomialNB` để so sánh.

---

## 3. Đối chiếu các yêu cầu bắt buộc

| Yêu cầu ban đầu | Implementation hiện tại | Đánh giá |
|---|---|---|
| Chọn một category, dùng file User Reviews | Dùng `All_Beauty.jsonl` thuộc Amazon Reviews 2023. | Đáp ứng |
| Item Metadata là tùy chọn | Không dùng metadata trong baseline text classification. | Đáp ứng |
| Phân loại theo `title + text` | `review_text` được ghép từ hai field sau khi chuẩn hóa khoảng trắng. | Đáp ứng |
| Giữ rating 1–5, loại các giá trị khác | Rating được chuyển sang `float` và chỉ map các giá trị 1, 2, 3, 4, 5. | Đáp ứng |
| Giữ record khi chỉ title hoặc text có nội dung | Chỉ loại khi cả hai field rỗng sau chuẩn hóa. | Đáp ứng |
| Loại record trùng | Dùng SHA-256 của JSON canonical; chỉ loại JSON record trùng hoàn toàn. | Đáp ứng, với tiêu chí cụ thể cần ghi rõ |
| Xuất `cleaned.jsonl` | Xuất tại `data/interim/cleaned.jsonl`. | Đáp ứng |
| Ghi số record bị loại | Có `preprocessing_stats.json`; audit cũng ghi đầy đủ số record theo từng quy tắc. | Đáp ứng |
| Tạo label từ rating | 1–2 → Negative, 3 → Neutral, 4–5 → Positive. | Đáp ứng |
| Ghi số lượng từng label | Negative: 143.489; Neutral: 55.720; Positive: 495.044. | Đáp ứng |
| Thêm `review_text` và `sentiment_class`, giữ field gốc | Record sạch sao chép từ record gốc và bổ sung hai field mới. | Đáp ứng |
| Xuất ARFF | ARFF có `review_text` và `sentiment_class`. | Đáp ứng |
| Khảo sát số record, độ dài review, phân bố rating | `eda_all_beauty.py` tạo các thống kê và biểu đồ. | Đáp ứng |

### 3.1. Kết quả dữ liệu đã xác nhận

| Chỉ số | Giá trị |
|---|---:|
| Tổng số dòng raw | 701.528 |
| Record JSON trùng hoàn toàn bị loại | 7.275 |
| Record sạch | 694.253 |
| Độ dài review trung bình | 36,67 từ |
| Rating 5 sao | 416.436 review, tương đương 59,98% |
| Lớp Positive | 495.044 review, tương đương 71,31% |

Audit hiện có xác nhận không có dòng rỗng, JSON lỗi, rating không hợp lệ, hoặc
record có cả `title` và `text` rỗng trong file raw đang dùng.

---

## 4. Những điểm cần diễn giải chính xác

### 4.1. Bài toán hai lớp hay ba lớp

Phần Objective ban đầu dùng cụm “tích cực hoặc tiêu cực”, nhưng quy tắc tạo nhãn
và phần Output lại nêu ba lớp Negative, Neutral, Positive. Code và toàn bộ
pipeline đã triển khai bài toán **ba lớp**, phù hợp với quy tắc chi tiết hơn.

Report cuối nên nêu rõ ngay từ đầu:

> Đây là bài toán phân loại cảm xúc ba lớp: Negative, Neutral và Positive.

Không nên mô tả project là bài toán nhị phân.

### 4.2. Tiêu chí duplicate

Code chỉ coi hai record là duplicate khi **toàn bộ JSON record** của chúng giống
nhau sau khi chuẩn hóa thứ tự key để hash. Vì vậy, hai review có cùng `title` và
`text` nhưng khác `user_id`, `rating`, timestamp hoặc field khác vẫn được giữ.

Đây là một lựa chọn bảo thủ và hợp lệ vì yêu cầu ban đầu không định nghĩa thế
nào là record trùng. Report cần công khai tiêu chí này; không cần đổi sang
dedupe theo text nếu không có yêu cầu mới.

### 4.3. Nội dung của file ARFF

`export_arff.py` xuất đúng hai attribute:

```arff
@ATTRIBUTE review_text STRING
@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}
```

Điều này đáp ứng yêu cầu xuất ARFF và phù hợp với bài toán text classification.
Yêu cầu ban đầu không bắt buộc đưa toàn bộ attribute gốc vào ARFF. Nếu cần dùng
ARFF cho một bài phân tích khác, có thể bổ sung field gốc, nhưng không được đưa
`rating` vào feature vì rating trực tiếp sinh ra label và sẽ gây target leakage.

### 4.4. Quy mô dữ liệu huấn luyện

`train_models.py` và `tune_models.py` mặc định lấy uniform reservoir sample
gồm 50.000 review. Mẫu này được chia phân tầng thành 40.000 review train và
10.000 review test.

Đây là một lựa chọn hợp lệ để chạy thí nghiệm nhanh, tái lập được và không phải
là lỗi so với yêu cầu ban đầu; yêu cầu không bắt buộc train toàn bộ dữ liệu.
Tuy nhiên, report phải nêu rõ kết quả hiện tại dựa trên mẫu 50.000 review. Có
thể chạy toàn bộ dữ liệu bằng:

```powershell
py src\train_models.py --max-records 0
```

Việc train toàn bộ nên được xem là hướng mở rộng hoặc kiểm chứng thêm, không
phải điều kiện bắt buộc để pipeline đạt yêu cầu.

---

## 5. Multinomial Naive Bayes và phần lý thuyết

### 5.1. Code hiện tại làm gì

`train_models.py` tạo feature bằng `TfidfVectorizer` (unigram + bigram,
tối đa 20.000 feature), sau đó dùng cùng ma trận TF-IDF cho Dummy Classifier,
Multinomial Naive Bayes, Logistic Regression và Linear SVM. `MultinomialNB()`
dùng `alpha=1.0` mặc định.

### 5.2. Đánh giá đúng mức

Phần lý thuyết ban đầu giải thích Multinomial Naive Bayes theo **word count**:
`x_i` là số lần từ xuất hiện, còn likelihood dùng `Count(w_i, c)`.

Vì code dùng TF-IDF thay cho word count, cấu hình hiện tại **khác với ví dụ lý
thuyết**. Tuy nhiên, đây không phải là lỗi thuật toán: `MultinomialNB` có thể
huấn luyện với các feature không âm như TF-IDF.

Do đó, không nên viết “NB không đúng công thức”. Cách diễn đạt chính xác là:

> Multinomial Naive Bayes hiện dùng TF-IDF để so sánh công bằng với các mô hình
> tuyến tính. Nếu muốn thực nghiệm bám sát phần lý thuyết word frequency, cần
> bổ sung một pipeline `CountVectorizer` + `MultinomialNB` để đối chiếu.

### 5.3. Cải tiến tùy chọn cho NB

Nếu giảng viên kỳ vọng demo trực tiếp công thức NB theo số lần từ xuất hiện,
có thể bổ sung các thí nghiệm sau:

1. `CountVectorizer(ngram_range=(1, 1))` + `MultinomialNB(alpha=1.0)`.
2. Thử một vài giá trị `alpha`, chẳng hạn 0,5; 1,0; 2,0, trên validation set.
3. So sánh Count-based NB với TF-IDF-based NB bằng Macro F1.

Đây là phần mở rộng nhằm liên kết code với lý thuyết tốt hơn; không làm mất giá
trị của baseline hiện tại.

---

## 6. Phần mở rộng ngoài yêu cầu ban đầu

Code đã bổ sung các thực hành tốt cho một project Data Mining:

| Nội dung | Implementation |
|---|---|
| Baseline tối thiểu | Dummy Classifier luôn dự đoán lớp phổ biến nhất |
| Mô hình bổ sung | Logistic Regression và Linear SVM |
| Tránh leakage | TF-IDF chỉ fit trên train set; rating không làm feature |
| Chia dữ liệu | Stratified train/test split 80/20, `random_state=42` |
| Metrics | Accuracy, Macro F1, Weighted F1, classification report, confusion matrix |
| Phân tích lỗi | Xuất các cặp nhầm lẫn và ví dụ review dự đoán sai |
| Tuning có kiểm soát | Validation được tách từ train; held-out test không được dùng để chọn cấu hình |

Các phần này là điểm mạnh của project, không phải phần lệch yêu cầu.

---

## 7. Model cuối và tính nhất quán của report

Các Phase 2 và 3 phân tích baseline chọn Logistic Regression vì nó có Macro F1
baseline cao nhất là 0,7406. Tuy nhiên, Phase 5 thực hiện tuning có validation
set đã chọn cấu hình sau:

| Thành phần | Giá trị |
|---|---|
| Model | Linear SVM |
| `C` | 0,5 |
| `class_weight` | `balanced` |
| TF-IDF | unigram + bigram, tối đa 20.000 feature |
| Tiêu chí chọn | Validation Macro F1 |
| Final test Accuracy | 0,8854 |
| Final test Macro F1 | 0,743034 |

Vì vậy, nếu report cuối bao gồm Phase 5, **Linear SVM với `C=0.5` phải là model
cuối được báo cáo**. Các đoạn trong `FINAL_REPORT_DRAFT.md` và report cũ vẫn
gọi Logistic Regression là “model chính” cần được cập nhật hoặc chú thích rõ
rằng đó là kết quả baseline trước tuning.

---

## 8. Tính nhất quán của artifact tiền xử lý

Source `preprocess_all_beauty.py` khởi tạo bảy bộ đếm, bao gồm cả các lý do có
giá trị 0. Tuy nhiên, file `data/interim/preprocessing_stats.json` hiện chỉ
hiển thị ba giá trị không bằng 0: số raw, số cleaned và số duplicate bị loại.

Trong khi đó, `data/interim/raw_data_audit.json` đang lưu đầy đủ các số liệu
zero cho dòng rỗng, JSON lỗi, rating lỗi và title/text cùng rỗng. Để artifact
nhất quán với source code và dễ đưa vào report, nên chạy lại preprocessing hoặc
kiểm tra lại phiên bản script đã dùng để tạo `preprocessing_stats.json`.

---

## 9. Danh sách việc cần làm theo mức ưu tiên

### Bắt buộc trước khi nộp

1. Cập nhật report tổng hợp để xác định đây là bài toán ba lớp.
2. Thống nhất Linear SVM `C=0.5` là model cuối sau tuning; phân biệt rõ với
   Logistic Regression của baseline.
3. Ghi rõ duplicate là duplicate JSON hoàn toàn.
4. Ghi rõ quy mô thí nghiệm: sample 50.000, train 40.000, held-out test 10.000.
5. Bảo đảm bảng thống kê preprocessing hiển thị đầy đủ các lý do loại record,
   kể cả khi giá trị bằng 0.

### Nên làm nếu còn thời gian hoặc nếu giảng viên yêu cầu

1. Bổ sung CountVectorizer + MultinomialNB để khớp trực tiếp phần lý thuyết.
2. Tune `alpha` của NB trên validation set.
3. Train lại trên toàn bộ 694.253 record sau khi chốt cấu hình, nếu tài nguyên
   cho phép.
4. Mở rộng ARFF với các field gốc chỉ khi có mục đích phân tích cụ thể; không
   dùng rating làm feature.

## 10. Kết luận

Project đã có pipeline hợp lý, tái lập được và vượt yêu cầu cơ bản ở các phần
EDA, đánh giá, phân tích lỗi, chống leakage và tuning. Phần cần cải thiện chính
là cách trình bày: thống nhất model cuối, mô tả rõ các lựa chọn thiết kế, và
liên kết thí nghiệm Naive Bayes với phần lý thuyết word-count khi cần.

