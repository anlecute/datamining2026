# Report Phase 5 — Tinh chỉnh có kiểm soát

## Project: Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu Phase 5

Phase 5 thử nghiệm một số cấu hình TF-IDF và model tuyến tính để cải thiện Macro
F1. Mỗi thí nghiệm thay đổi một yếu tố chính và dùng cùng sample, cùng seed,
cùng validation set. Không dùng BERT vì baseline TF-IDF kết hợp model tuyến tính
đã phù hợp với phạm vi Data Mining cơ bản, dễ giải thích và ít đòi hỏi tài nguyên.

Điểm quan trọng là không dùng test set để chọn cấu hình. Test set chỉ được dùng
một lần sau khi configuration thắng đã được chọn bằng validation Macro F1.

## 2. Thiết kế thí nghiệm đáng tin cậy

### 2.1. Chia dữ liệu

Từ cùng sample 50.000 review với random_state=42, dữ liệu được chia thành ba
phần:

| Tập | Số review | Vai trò |
|---|---:|---|
| Fit set | 32.000 | Fit TF-IDF và train từng configuration |
| Validation set | 8.000 | So sánh configuration theo Macro F1 |
| Held-out test set | 10.000 | Đánh giá cuối cùng đúng một lần |

Quy trình là lấy outer train/test 80/20 trước, sau đó tách 20% của outer train
làm validation. Tất cả split đều dùng stratified split và seed 42 để giữ tỷ lệ
ba class và có thể tái lập.

### 2.2. Quy tắc chọn cấu hình

1. Mỗi configuration được train trên 32.000 review fit set.
2. Mỗi configuration được đánh giá trên cùng 8.000 review validation set.
3. Configuration có validation Macro F1 cao nhất được chọn.
4. Configuration thắng được train lại trên 40.000 outer-train review.
5. Model cuối cùng được đánh giá một lần trên 10.000 held-out test review.

Nhờ quy trình này, test set không ảnh hưởng việc chọn n-gram, class weight,
vocabulary size hoặc C.

## 3. Các thí nghiệm được thực hiện

| Nhóm thử nghiệm | Cấu hình được so sánh | Câu hỏi |
|---|---|---|
| N-gram | Logistic Regression unigram (1,1) và unigram+bigram (1,2) | Bigram có giúp nhận diện ngữ cảnh như not good không? |
| Class weight | Logistic Regression không weight và balanced | Macro F1/Neutral F1 có cải thiện không? |
| Vocabulary size | 10.000 và 20.000 features | Vocabulary lớn hơn có hữu ích không? |
| Logistic Regression C | 0,5; 1,0; 2,0 | Regularization ảnh hưởng thế nào? |
| Linear SVM C | 0,5; 1,0; 2,0 | C nào cho Macro F1 tốt nhất? |

Ngoại trừ yếu tố đang kiểm tra, Logistic Regression baseline giữ bigram (1,2),
20.000 feature, class_weight=balanced và C=1,0. Linear SVM giữ bigram, 20.000
feature và class_weight=balanced; chỉ thay đổi C.

## 4. Kết quả trên validation set

Nguồn số liệu: outputs/metrics/tuning_results.csv.

| Thí nghiệm | Model | N-gram | Feature | Weight | C | Accuracy | Macro F1 | Neutral F1 |
|---|---|---|---:|---|---:|---:|---:|---:|
| lr_baseline | Logistic Regression | (1,2) | 20.000 | balanced | 1,0 | 0,8513 | 0,7233 | 0,4386 |
| lr_unigram | Logistic Regression | (1,1) | 20.000 | balanced | 1,0 | 0,8335 | 0,7095 | 0,4265 |
| lr_no_class_weight | Logistic Regression | (1,2) | 20.000 | none | 1,0 | 0,8824 | 0,6926 | 0,3103 |
| lr_10k_features | Logistic Regression | (1,2) | 10.000 | balanced | 1,0 | 0,8436 | 0,7159 | 0,4277 |
| lr_c_0_5 | Logistic Regression | (1,2) | 20.000 | balanced | 0,5 | 0,8441 | 0,7211 | 0,4411 |
| lr_c_2_0 | Logistic Regression | (1,2) | 20.000 | balanced | 2,0 | 0,8566 | 0,7224 | 0,4295 |
| svm_c_0_5 | Linear SVM | (1,2) | 20.000 | balanced | 0,5 | 0,8774 | **0,7312** | 0,4281 |
| svm_c_1_0 | Linear SVM | (1,2) | 20.000 | balanced | 1,0 | 0,8739 | 0,7208 | 0,4051 |
| svm_c_2_0 | Linear SVM | (1,2) | 20.000 | balanced | 2,0 | 0,8683 | 0,7107 | 0,3875 |

## 5. Diễn giải từng thí nghiệm

### 5.1. Unigram và bigram

Logistic Regression với bigram đạt Macro F1 0,7233, cao hơn unigram 0,7095;
chênh lệch là 0,0138. Neutral F1 cũng tăng từ 0,4265 lên 0,4386. Kết quả ủng hộ
việc dùng bigram vì các cụm hai từ có thể giữ thông tin phủ định/đối lập tốt hơn
một từ riêng lẻ, ví dụ not good hoặc never again.

### 5.2. Ảnh hưởng của class weight

Cấu hình không dùng class weight có Accuracy 0,8824, cao hơn cấu hình balanced
(0,8513), nhưng Macro F1 giảm từ 0,7233 xuống 0,6926. Neutral F1 giảm mạnh từ
0,4386 xuống 0,3103. Điều này cho thấy Accuracy có thể ưu ái lớp Positive đa
số; class_weight=balanced giúp model quan tâm nhiều hơn đến Neutral và Negative.

### 5.3. Số lượng feature

20.000 feature đạt Macro F1 0,7233, cao hơn 10.000 feature (0,7159). Vocabulary
lớn hơn giúp giữ thêm các từ/cụm từ có ích cho sentiment classification. Mức
cải thiện là 0,0074, không lớn nhưng nhất quán trong thí nghiệm này.

### 5.4. Regularization của Logistic Regression

Với Logistic Regression, C=1,0 đạt Macro F1 cao nhất (0,7233), sau đó là C=2,0
(0,7224) và C=0,5 (0,7211). C=0,5 có Neutral F1 cao nhất (0,4411), nhưng tiêu
chí chọn chính là Macro F1 trên cả ba class, nên C=1,0 là lựa chọn Logistic
Regression tốt nhất trong các cấu hình thử.

### 5.5. Regularization của Linear SVM

Linear SVM với C=0,5 đạt Macro F1 0,7312, cao hơn C=1,0 (0,7208) và C=2,0
(0,7107). Khi tăng C, model bị regularization ít hơn nhưng Macro F1 và Neutral
F1 đều giảm. Vì vậy C=0,5 được chọn là cấu hình tốt nhất trên validation set.

## 6. Đánh giá cuối trên held-out test set

Sau khi chọn svm_c_0_5 bằng validation Macro F1, model này được train lại trên
40.000 outer-train review và đánh giá đúng một lần trên 10.000 held-out test
review.

| Metric | Kết quả cuối |
|---|---:|
| Accuracy | 0,8854 |
| Macro F1 | **0,7430** |
| Weighted F1 | 0,8841 |
| Negative F1 | 0,8273 |
| Neutral F1 | 0,4514 |
| Positive F1 | 0,9504 |

Trước tuning, Linear SVM mặc định C=1,0 có Macro F1 0,7353 trên cùng held-out
test setup. Cấu hình C=0,5 nâng Macro F1 lên 0,7430, cải thiện 0,0077; Accuracy
tăng từ 0,8814 lên 0,8854. Điều này xác nhận configuration được chọn trên
validation vẫn hoạt động tốt trên test set chưa được dùng trong selection.

## 7. Cấu hình được chọn

| Thành phần | Giá trị |
|---|---|
| Model | Linear SVM |
| C | 0,5 |
| class_weight | balanced |
| TF-IDF n-gram | (1,2) |
| max_features | 20.000 |
| min_df | 2 |
| max_df | 0,95 |
| Tiêu chí chọn | Validation Macro F1 |

Các artifact tái lập là:

    outputs/metrics/tuning_configuration.json
    outputs/metrics/tuning_results.csv
    outputs/metrics/tuning_results.json
    outputs/metrics/tuning_final_test_result.json

## 8. Kết luận Phase 5

Phase 5 hoàn thành tiêu chí tinh chỉnh có kiểm soát:

- Có chín thí nghiệm nhỏ và một bảng kết quả chung.
- Mỗi nhóm thử nghiệm chỉ thay đổi một yếu tố chính.
- Toàn bộ configuration dùng cùng sample, seed và validation set.
- Test set được giữ riêng đến khi chọn xong configuration.
- Linear SVM C=0,5 được chọn theo validation Macro F1, không theo Accuracy đơn lẻ.
- Kết quả cuối trên held-out test là Macro F1 0,7430, cao hơn Linear SVM mặc
  định và cao hơn Logistic Regression baseline trước tuning (Macro F1 0,7406).

Tuning cải thiện kết quả ở mức nhỏ nhưng có kiểm soát. Neutral vẫn là class có
F1 thấp nhất, vì vậy giới hạn do nhãn rating và review cảm xúc pha trộn vẫn cần
được nêu trong report tổng kết.
