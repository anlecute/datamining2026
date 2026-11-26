# Kế hoạch thực hiện project Data Mining

## Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu và phạm vi đã chốt

**Mục tiêu:** dự đoán review thuộc một trong ba lớp `Negative`, `Neutral`,
`Positive` từ nội dung `title + text`.

**Dataset:** Amazon Reviews 2023, category `All_Beauty`, chỉ dùng file
**User Reviews**. Item Metadata không dùng trong baseline vì không chứa cảm xúc
của người viết review.

**Quy tắc nhãn:**

| Rating | Label |
|---:|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

**Tiêu chí chọn mô hình:** Macro F1, vì lớp Positive chiếm đa số còn Neutral
ít mẫu hơn.

---

## 2. Trạng thái hiện tại

| Hạng mục cô yêu cầu | Trạng thái | Bằng chứng/đầu ra |
|---|---|---|
| Chọn category và file User Reviews | Hoàn thành | `All_Beauty.jsonl` |
| Kiểm tra dữ liệu raw | Hoàn thành | `data/interim/raw_data_audit.json` |
| Lọc dữ liệu, xử lý duplicate | Hoàn thành | `data/interim/cleaned.jsonl` |
| Tạo `review_text` và `sentiment_class` | Hoàn thành | `cleaned.jsonl` |
| Ghi số liệu record bị loại | Hoàn thành | `preprocessing_stats.json` và audit |
| Xuất ARFF | Hoàn thành | `data/processed/amazon_sentiment.arff` |
| Chạy baseline models | Hoàn thành | `outputs/metrics/model_comparison.csv` |
| EDA và biểu đồ dữ liệu | Chưa làm | Việc ưu tiên tiếp theo |
| Phân tích lỗi model | Chưa làm | Làm sau EDA |
| Tinh chỉnh có kiểm soát | Chưa làm | Làm sau khi có baseline/EDA |
| Hoàn thiện report và slide | Đang thực hiện | Bổ sung theo từng phase |

Kết quả baseline hiện có: Logistic Regression có Macro F1 cao nhất là 0,7406;
Linear SVM có Accuracy cao nhất là 0,8814. Vì dữ liệu mất cân bằng, sẽ chọn
Logistic Regression làm mô hình chính để phân tích, nhưng vẫn báo cáo cả bốn
mô hình.

---

## 3. Roadmap thực hiện

### Phase 1 — EDA và trực quan hóa dữ liệu

**Mục tiêu:** trả lời bằng số liệu và biểu đồ rằng dataset có đặc điểm gì trước
khi huấn luyện model.

| Việc thực hiện | Câu hỏi cần trả lời | Đầu ra cần nộp |
|---|---|---|
| Đếm record raw và cleaned | Đã loại bao nhiêu record, vì sao? | Bảng cleaning summary |
| Đếm rating 1–5 | Rating nào phổ biến nhất? | Bar chart rating |
| Đếm 3 sentiment classes | Dữ liệu có mất cân bằng không? | Bar chart sentiment |
| Tính số từ của `review_text` | Review trung bình dài bao nhiêu từ? | Bảng mean, median, min, max |
| So sánh độ dài theo class | Negative/Neutral/Positive có độ dài khác nhau không? | Bảng và boxplot/bar chart |
| Kiểm tra top words tùy chọn | Từ nào xuất hiện nhiều theo lớp? | Bảng top words, phần thảo luận |

**Điều cần học song song:**

- Pandas: `DataFrame`, `value_counts`, `groupby`, `describe`.
- Thống kê mô tả: count, percentage, mean, median, min, max.
- Matplotlib: bar chart, histogram, boxplot.
- Ý nghĩa của class imbalance.

**Tiêu chí hoàn thành:** có ít nhất 3 biểu đồ, 2 bảng thống kê, và mỗi biểu đồ
có đoạn diễn giải 2–3 câu để đưa vào report.

### Phase 2 — Xác nhận phương pháp mô hình hóa

**Mục tiêu:** giải thích rõ mô hình học từ đâu và vì sao quy trình đáng tin cậy.

| Nội dung | Cần trình bày trong report |
|---|---|
| Train/test split | Chia 80/20 và dùng stratified split |
| Tái lập | `random_state=42` |
| Feature extraction | TF-IDF, unigram + bigram |
| Chống data leakage | Chỉ fit TF-IDF trên train set |
| Baseline | Dummy Classifier để so sánh tối thiểu |
| Models | Naive Bayes, Logistic Regression, Linear SVM |
| Mất cân bằng lớp | Dùng Macro F1 và `class_weight="balanced"` |

**Điều cần học song song:** supervised learning, multi-class classification,
feature/label, train/test split, overfitting, data leakage, TF-IDF và n-gram.

**Tiêu chí hoàn thành:** bạn giải thích được vì sao không dùng `rating` làm
feature và vì sao Logistic Regression được chọn theo Macro F1 thay vì chỉ nhìn
Accuracy.

### Phase 3 — Phân tích kết quả baseline

**Mục tiêu:** biến bảng metric thành nhận xét có ý nghĩa.

| Việc thực hiện | Cách làm | Đầu ra |
|---|---|---|
| So sánh 4 models | Dùng Accuracy, Macro F1, Weighted F1 | Bảng model comparison |
| Đọc classification report | So sánh precision/recall/F1 theo lớp | Bảng chi tiết theo class |
| Đọc confusion matrix | Xác định các cặp class bị nhầm | Hình confusion matrix + nhận xét |
| Giải thích lựa chọn | Ưu tiên Macro F1 do mất cân bằng | Kết luận mô hình chính |

**Nhận xét dự kiến cần kiểm chứng từ kết quả hiện có:** lớp Neutral khó nhất
(F1 khoảng 0,47), vì review 3 sao thường có cảm xúc pha trộn hoặc không rõ rệt.

**Tiêu chí hoàn thành:** có 1 bảng so sánh, 1 confusion matrix và ít nhất 3
nhận xét dựa trực tiếp trên số liệu.

### Phase 4 — Phân tích lỗi

**Mục tiêu:** không chỉ nói model sai, mà chỉ ra model sai trong trường hợp nào.

| Việc thực hiện | Số lượng gợi ý | Câu hỏi phân tích |
|---|---:|---|
| Lấy review dự đoán sai | 20–30 review | Actual và predicted khác nhau thế nào? |
| Chia lỗi theo cặp class | 5–10 review cho mỗi cặp chính | Neutral bị nhầm sang Positive/Negative ra sao? |
| Đọc text gốc | Kiểm tra từ phủ định, câu mơ hồ, nhãn rating | Vì sao label rating và nội dung có thể không khớp? |
| Viết kết luận lỗi | 3–5 nhóm lỗi | Hạn chế của model và dataset |

Ví dụ nhóm lỗi có thể gặp:

1. Review 3 sao vừa khen vừa chê nên khó là Neutral.
2. Câu quá ngắn như “Okay” hoặc “Not bad” thiếu ngữ cảnh.
3. Từ tích cực đi cùng phủ định, ví dụ “not good”.
4. Rating và câu viết không cùng mức cảm xúc.
5. Review có thông tin sản phẩm nhưng ít bộc lộ cảm xúc.

**Tiêu chí hoàn thành:** có bảng 10–20 ví dụ sai đại diện, không cần công bố
toàn bộ review test set.

### Phase 5 — Tinh chỉnh có kiểm soát

**Mục tiêu:** thử nghiệm nhỏ, có câu hỏi rõ ràng và không làm thay đổi test set.

Thử lần lượt, mỗi lần chỉ thay đổi một yếu tố:

| Thử nghiệm | Cấu hình A | Cấu hình B | Câu hỏi |
|---|---|---|---|
| N-gram | `(1, 1)` | `(1, 2)` | Bigram có giúp nhận diện phủ định không? |
| Class weight | Không dùng | `balanced` | Macro F1 Neutral có cải thiện không? |
| Số feature | 10.000 | 20.000 | Vocabulary lớn hơn có hữu ích không? |
| Logistic Regression C | 0,5 | 1,0 hoặc 2,0 | Regularization ảnh hưởng thế nào? |
| Linear SVM C | 0,5 | 1,0 hoặc 2,0 | Có tăng Macro F1 không? |

Không thử BERT ở phase này. BERT cần nhiều tài nguyên và làm project khó giải
thích hơn; TF-IDF + các mô hình tuyến tính đã phù hợp với mục tiêu Data Mining
cơ bản.

**Tiêu chí hoàn thành:** có một bảng thí nghiệm nhỏ; chọn cấu hình tốt nhất
theo Macro F1, không theo Accuracy đơn lẻ.

### Phase 6 — Hoàn thiện report và slide

**Mục tiêu:** chuyển toàn bộ kết quả thành bài nộp mạch lạc, có bằng chứng.

| Phần report | Nội dung cần có | Artifact hiện có/cần tạo |
|---|---|---|
| Giới thiệu | Bài toán, mục tiêu, phạm vi | `report_16102026.md` |
| Dataset | Nguồn, category, schema, lý do chọn User Reviews | Audit + mô tả dataset |
| Tiền xử lý | Quy tắc lọc, label, `review_text`, duplicate | Audit, stats, pseudocode |
| EDA | Bảng, biểu đồ và nhận xét | Cần tạo ở Phase 1 |
| Phương pháp | Split, TF-IDF, mô hình, metrics | Source code + mô tả |
| Kết quả | So sánh model và confusion matrix | Metrics/figure hiện có |
| Thảo luận | Lỗi, class imbalance, giới hạn rating-label | Cần tạo ở Phase 4 |
| Kết luận | Mô hình tốt nhất, giới hạn, hướng phát triển | Viết cuối cùng |

Slide thuyết trình nên có khoảng 8–10 slide: bài toán, dữ liệu, tiền xử lý,
EDA, phương pháp, kết quả, confusion matrix, lỗi/hạn chế, kết luận.

---

## 4. Lịch thực hiện đề xuất trong 10 buổi

| Buổi | Công việc chính | Kết quả cuối buổi |
|---:|---|---|
| 1 | Học Pandas cơ bản; đọc `cleaned.jsonl` | Hiểu cột và số record |
| 2 | Thống kê rating, label, độ dài review | Bảng EDA đầu tiên |
| 3 | Vẽ biểu đồ rating, label, độ dài | 3 biểu đồ lưu vào `outputs/figures` |
| 4 | Viết phần Dataset + EDA trong report | Hoàn thành chương dữ liệu |
| 5 | Học TF-IDF, split, leakage, metrics | Giải thích được pipeline |
| 6 | Đọc model comparison và confusion matrix | Hoàn thành nhận xét kết quả |
| 7 | Xuất và đọc mẫu dự đoán sai | Bảng error analysis |
| 8 | Thử 2–4 cấu hình nhỏ | Bảng tuning experiment |
| 9 | Viết thảo luận, giới hạn và kết luận | Report gần hoàn chỉnh |
| 10 | Rà soát số liệu, chuẩn bị slide và luyện trình bày | Bản nộp cuối |

---

## 5. Checklist trước khi nộp

### Dataset và tiền xử lý

- [x] Chọn một category rõ ràng: All_Beauty.
- [x] Dùng đúng file User Reviews.
- [x] Kiểm tra toàn bộ raw data.
- [x] Có số record bị loại theo từng lý do.
- [x] Giữ title hoặc text nếu một trong hai có nội dung.
- [x] Loại duplicate hoàn toàn.
- [x] Tạo `review_text` và `sentiment_class`.
- [x] Xuất `cleaned.jsonl` và ARFF.

### EDA và trực quan hóa

- [ ] Có thống kê số record raw/cleaned.
- [ ] Có phân bố rating 1–5.
- [ ] Có phân bố ba sentiment classes.
- [ ] Có độ dài review trung bình/median.
- [ ] Có ít nhất ba biểu đồ và nhận xét tương ứng.

### Mô hình

- [x] Có stratified train/test split.
- [x] Không có leakage khi fit TF-IDF.
- [x] Có Dummy baseline.
- [x] Có Naive Bayes, Logistic Regression, Linear SVM.
- [x] Có Accuracy, Macro F1, Weighted F1.
- [x] Có confusion matrix.
- [ ] Có bảng thử nghiệm tinh chỉnh nhỏ.
- [ ] Có phân tích 10–20 dự đoán sai.

### Bài nộp

- [ ] EDA và biểu đồ đã chèn vào report.
- [ ] Kết quả được giải thích, không chỉ đưa bảng số.
- [ ] Nêu rõ Neutral là lớp khó và lý do có thể xảy ra.
- [ ] Nêu giới hạn: nhãn suy ra từ rating không phải sentiment label gán tay.
- [ ] Kiểm tra mọi số liệu trong report trùng với artifact JSON/CSV.

---

## 6. Việc làm ngay bây giờ

**Bước tiếp theo duy nhất:** thực hiện Phase 1 — EDA và trực quan hóa.

Mục tiêu cụ thể của bước này là tạo một script EDA xuất các bảng thống kê và
ít nhất ba biểu đồ: phân bố rating, phân bố sentiment class và phân bố độ dài
review. Sau khi có EDA, mới chuyển sang phân tích lỗi và tuning model.
