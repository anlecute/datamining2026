# Report Phase 4 — Phân tích lỗi mô hình

## Project: Phân loại cảm xúc Amazon All_Beauty

## 1. Mục tiêu Phase 4

Phase 4 phân tích các trường hợp mà Logistic Regression dự đoán sai trên test
set. Mục tiêu không chỉ nêu số lỗi, mà xác định model sai ở những tình huống
nào, giải thích nguyên nhân có thể, và chỉ ra giới hạn của dataset cũng như
phương pháp hiện tại.

Mô hình được phân tích là Logistic Regression vì có Macro F1 cao nhất (0,7406).
Phân tích dùng các file:

| File | Vai trò |
|---|---|
| `test_predictions_best_model.csv` | 10.000 dự đoán test, có nhãn thật, nhãn dự đoán, rating gốc và text |
| `error_analysis_summary.json` | Số lỗi theo cặp class |
| `error_examples.csv` | 30 ví dụ lỗi, 5 ví dụ cho mỗi cặp nhầm lẫn |

Các ví dụ được chọn bằng random seed 42. Báo cáo chọn 18 review đại diện, ba
review cho mỗi cặp nhầm lẫn. Đây là phân tích định tính minh họa; không dùng 18
ví dụ để suy ra tỷ lệ của toàn bộ dataset.

## 2. Tổng quan lỗi trên test set

| Chỉ số | Kết quả |
|---|---:|
| Tổng review test | 10.000 |
| Dự đoán đúng | 8.632 |
| Dự đoán sai | 1.368 |
| Accuracy | 0,8632 |
| Error rate | 13,68% |

Sáu cặp nhầm lẫn có số lỗi như sau:

| Nhãn thật | Dự đoán | Số lỗi |
|---|---|---:|
| Positive | Neutral | 481 |
| Negative | Neutral | 278 |
| Neutral | Negative | 209 |
| Positive | Negative | 188 |
| Neutral | Positive | 128 |
| Negative | Positive | 84 |

Neutral xuất hiện ở nhiều hướng nhầm lẫn: Positive → Neutral, Negative →
Neutral, Neutral → Negative và Neutral → Positive. Đây là bằng chứng định lượng
cho thấy ranh giới của Neutral không rõ ràng bằng hai lớp còn lại.

## 3. Các nhóm nguyên nhân lỗi

Từ 18 ví dụ đại diện, có năm nhóm nguyên nhân chính.

| Nhóm lỗi | Mô tả |
|---|---|
| Cảm xúc pha trộn | Review có cả lời khen và phàn nàn; model khó xác định cảm xúc tổng thể. |
| Từ phủ định/đối lập | Các cụm như `not good`, `never`, `but`, `however` làm thay đổi nghĩa của từ tích cực. |
| Kỳ vọng không đạt | Review nêu sản phẩm vẫn dùng được nhưng không giống ảnh, không đúng texture hoặc không đủ hiệu quả. |
| Nhãn rating và text lệch nhau | Người dùng đánh 1–2 hoặc 4–5 sao nhưng text lại chứa nhiều tín hiệu cảm xúc ngược lại. |
| Câu mơ hồ hoặc thiên về mô tả | Text chỉ mô tả đặc tính sản phẩm, ít cho biết mức độ hài lòng tổng thể. |

## 4. Bảng 18 ví dụ lỗi đại diện

Nội dung dưới đây là tóm tắt ngắn từ review gốc trong `error_examples.csv`;
rating gốc được giữ để kiểm tra mối quan hệ giữa text và label.

| # | Rating | Nhãn thật | Dự đoán | Dấu hiệu chính trong review | Nhóm lỗi/diễn giải |
|---:|---:|---|---|---|---|
| 1 | 1 | Negative | Neutral | Hair ties “too small”, nhưng “may be okay” cho kiểu tóc khác | Có phàn nàn chính kèm một khả năng sử dụng tích cực nhỏ. |
| 2 | 2 | Negative | Neutral | Bàn chải “priced nice” nhưng “wouldn't recommend” | Giá tốt và lời chê xuất hiện cùng nhau, khiến tín hiệu không thuần tiêu cực. |
| 3 | 1 | Negative | Neutral | “Not wood at all” nhưng “build quality was ok” | Review có chỉ trích sai mô tả sản phẩm nhưng vẫn khen một vài chi tiết. |
| 4 | 2 | Negative | Positive | Công thức mới gây khô tóc, nhưng từng “amazing” trước đây | Cảm xúc tích cực lịch sử lấn át phàn nàn về phiên bản hiện tại. |
| 5 | 1 | Negative | Positive | “Wanted to love”, thành phần tốt nhưng làm tóc xơ | Từ/cụm từ tích cực xuất hiện cùng kết luận tiêu cực. |
| 6 | 1 | Negative | Positive | Giao hàng đúng hẹn, màu phù hợp, nhưng pump bị lỗi | Khen giao hàng/màu sắc làm model bỏ qua lỗi chức năng chính. |
| 7 | 3 | Neutral | Negative | Quà không giống kỳ vọng, phải chỉnh lại hộp | Nội dung nghiêng tiêu cực, dù rating vẫn là 3 sao. |
| 8 | 3 | Neutral | Negative | Son dưỡng “does not moisturize”, phải bôi liên tục | Tín hiệu ngôn ngữ tiêu cực rất mạnh so với nhãn Neutral từ rating. |
| 9 | 3 | Neutral | Negative | Kẹp tóc bền nhưng “doesn't look like photo” | Khen và chê; phần chê về kỳ vọng dễ làm model đoán Negative. |
| 10 | 3 | Neutral | Positive | Mùi dịu, “still very pleased”, nhưng không đủ thơm như mong muốn | Lời khen rõ ràng làm model đoán Positive dù có một điểm chưa hài lòng. |
| 11 | 3 | Neutral | Positive | Dùng nhiều năm, hài lòng nhưng giá hiện tại quá cao | Cảm xúc tổng thể tích cực và phàn nàn về giá cùng tồn tại. |
| 12 | 3 | Neutral | Positive | “Awesome exfoliation” nhưng sản phẩm bị mô tả sai | Từ “awesome” là tín hiệu Positive mạnh trong TF-IDF. |
| 13 | 4 | Positive | Negative | Người dùng làm đổ dung dịch do quen chai nhỏ | Từ “spill” và mô tả sự cố không phản ánh mức hài lòng tổng thể. |
| 14 | 5 | Positive | Negative | Mascara yêu thích nhưng “not easy” để tìm mua | Model có thể bị ảnh hưởng bởi phủ định “not”/“never” dù review khen sản phẩm. |
| 15 | 5 | Positive | Negative | “Not worth it” với lần mua đầu, nhưng thay thế sau đó tốt hơn | Phàn nàn mạnh ở đầu review lấn át kết luận tích cực sau đó. |
| 16 | 4 | Positive | Neutral | Thiếu lớp cao su giữ kẹp, nhưng vẫn giữ tóc tốt | Review thiên về mô tả tính năng, mức cảm xúc không mạnh. |
| 17 | 5 | Positive | Neutral | Khó dùng nhưng “it does work” khi dùng đúng hướng dẫn | Câu chứa trade-off giữa khó dùng và hiệu quả. |
| 18 | 4 | Positive | Neutral | Mùi thơm, tẩy da chết tốt nhưng texture không như kỳ vọng | Cảm xúc pha trộn làm model chuyển về Neutral. |

## 5. Phân tích theo từng cặp class

### 5.1. Negative → Neutral và Negative → Positive

Negative → Neutral có 278 lỗi, còn Negative → Positive có 84 lỗi. Nhiều review
Negative vẫn có các cụm từ tích cực như “liked the size”, “build quality was
ok”, “amazing” hoặc lời khen về giao hàng/màu sắc. Đối với model TF-IDF, các
term tích cực này có thể bù trừ hoặc lấn át phần phàn nàn chính, đặc biệt khi
review mô tả nhiều khía cạnh sản phẩm.

### 5.2. Neutral → Negative và Neutral → Positive

Toàn bộ review Neutral có rating gốc 3 sao. Tuy nhiên, một số review 3 sao có
nội dung gần như chỉ chê (“does not moisturize”, “doesn't look like photo”), nên
model dự đoán Negative. Ngược lại, một số review dùng từ rất tích cực như
“awesome”, “very pleased” hoặc “happy”, nhưng vẫn nêu một hạn chế nhỏ và nhận
rating 3 sao. Vì nhãn được suy ra từ rating, text và class không luôn hoàn toàn
đồng nhất.

### 5.3. Positive → Negative và Positive → Neutral

Positive → Neutral là cặp lỗi lớn nhất với 481 review. Nhiều review 4–5 sao vẫn
góp ý về thao tác, texture, tính năng hoặc kỳ vọng. Positive → Negative thường
xảy ra khi review chứa từ phủ định mạnh hoặc kể lại một trải nghiệm tệ ở một
thời điểm, dù kết luận tổng thể vẫn hài lòng. Điều này cho thấy n-gram `(1, 2)`
giúp một phần với phủ định nhưng chưa thể nắm đầy đủ cấu trúc câu dài, thứ tự
thông tin hoặc ngữ cảnh thời gian.

## 6. Hạn chế của dữ liệu và model

1. **Label proxy từ rating:** sentiment class không phải nhãn do người đọc text
   gán. Rating 3 sao có thể có ngôn ngữ rất tích cực hoặc rất tiêu cực.
2. **Bag-of-words/TF-IDF hạn chế ngữ cảnh:** model xem term và n-gram, nhưng
   không hiểu đầy đủ quan hệ như “dùng tốt sau khi được thay thế”.
3. **Cảm xúc theo nhiều khía cạnh:** một review có thể khen màu sắc nhưng chê
   pump, hoặc khen hiệu quả nhưng chê giá. Mô hình chỉ dự đoán một nhãn tổng.
4. **Mất cân bằng lớp:** Neutral có ít mẫu nhất nên ít ví dụ huấn luyện hơn.
5. **Review rất ngắn hoặc mơ hồ:** các câu như “It's ok” hoặc mô tả chức năng
   có ít thông tin cho việc suy ra cảm xúc.

## 7. Kết luận Phase 4

Phase 4 đã đáp ứng tiêu chí yêu cầu:

- Có 30 review dự đoán sai được xuất ra, trong đó report phân tích 18 review
  đại diện.
- Mỗi trong sáu cặp nhầm lẫn có ba ví dụ được trình bày.
- Rating gốc, nhãn thật, nhãn dự đoán và nội dung review đều được đối chiếu.
- Năm nhóm nguyên nhân lỗi được xác định.
- Hạn chế của nhãn tạo từ rating và TF-IDF đã được nêu rõ.

Kết quả cho thấy việc cải thiện model không chỉ là tăng Accuracy. Cần đặc biệt
tập trung vào Neutral, review cảm xúc pha trộn và các cấu trúc phủ định/đối lập.
Phase tiếp theo là tinh chỉnh có kiểm soát: thay đổi từng tham số TF-IDF hoặc
model và so sánh Macro F1 trên cùng train/test split.
