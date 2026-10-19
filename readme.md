# Tài liệu học và triển khai project phân loại cảm xúc Amazon Reviews

## 1. Tổng quan bài toán

### 1.1. Bài toán cần giải quyết

Mục tiêu của project là xây dựng một mô hình có thể đọc nội dung review Amazon và dự đoán cảm xúc của người dùng thuộc một trong ba nhóm:

- `Negative`: tiêu cực
- `Neutral`: trung tính
- `Positive`: tích cực

Dữ liệu đầu vào chính là nội dung đánh giá gồm:

- `title`: tiêu đề ngắn của review
- `text`: nội dung chi tiết của review

Hai trường này sẽ được ghép thành một trường mới:

```text
review_text = title + " " + text
```

Ví dụ:

```text
title: "Battery died"
text: "It stopped working after two days."

review_text:
"Battery died It stopped working after two days."
```

Đầu ra mong muốn của mô hình:

```text
Input:  "The product broke after one day. Very disappointed."
Output: Negative
```

### 1.2. Bản chất của bài toán

Đây là bài toán thuộc nhóm:

- Data Mining
- Machine Learning
- Natural Language Processing (NLP)
- Supervised Learning
- Multi-class Classification

Nó là **supervised learning** vì dữ liệu huấn luyện đã có nhãn. Tuy nhiên, nhãn không có sẵn trực tiếp mà được tạo từ trường `rating`.

Quy tắc tạo nhãn:

| Rating | Sentiment class |
|---:|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

Ví dụ:

| Rating | Review | Nhãn |
|---:|---|---|
| 1 | “It stopped working.” | Negative |
| 2 | “Not worth the money.” | Negative |
| 3 | “It is okay, but nothing special.” | Neutral |
| 4 | “Works well for the price.” | Positive |
| 5 | “Excellent product. Highly recommended!” | Positive |

---

# 2. Sử dụng dataset Amazon Reviews 2023 cho project

## 2.1. Chọn đúng loại file

Nguồn chính thức: [Amazon Reviews'23 — McAuley Lab](https://amazon-reviews-2023.github.io/). Dataset gồm ba nhóm dữ liệu: **User Reviews**, **Item Metadata** và các file chỉ chứa tương tác/ID. Project này là phân loại cảm xúc từ văn bản review, vì vậy chỉ dùng **User Reviews**.

| Loại file | Có `rating`, `title`, `text`? | Dùng cho project này? | Mục đích chính |
|---|---|---|---|
| `review` / `raw_review_<Category>` | Có | **Có — bắt buộc** | Phân loại cảm xúc từ nội dung review |
| `meta` / `raw_meta_<Category>` | Không có review text | Tùy chọn | Bổ sung thông tin sản phẩm qua `parent_asin` |
| `rating_only` / `0core_rating_only_<Category>` | Có rating nhưng không có `title`, `text` | Không | Recommendation/collaborative filtering |
| `timestamp_split` | Có user, item, rating, thời gian nhưng không có review text | Không | Sequential recommendation |

> **Không dùng file `rating_only` hoặc `timestamp_split` để train sentiment model.** Chúng không chứa `title` và `text`, nên không thể tạo biến đầu vào `review_text`.

## 2.2. File đang dùng trong project này

Project sử dụng trực tiếp file **User Reviews đã giải nén** tại:

```text
All_Beauty.jsonl\All_Beauty.jsonl
```

Đây là file raw review của category `All_Beauty`. File `.jsonl.gz` chỉ là bản nén tải từ nguồn chính thức; không dùng bản nén trong các ví dụ bên dưới vì bản `.jsonl` đã giải nén đã có sẵn.

Mỗi dòng trong file là **một JSON object độc lập**, tương ứng một lượt đánh giá. Không mở hoặc chỉnh sửa file bằng Excel. Khi xử lý, phải đọc từng dòng; `json.load()` không phù hợp vì file không phải một JSON array duy nhất.

## 2.3. Đọc và kiểm tra file hiện tại

### Bước 1 — khai báo đường dẫn và đọc record đầu tiên

Trong Python, thêm tiền tố `r` trước chuỗi đường dẫn Windows để ký tự `\` được giữ nguyên.

```python
from pathlib import Path
import json

INPUT_PATH = Path(r"All_Beauty.jsonl\All_Beauty.jsonl")

if not INPUT_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy file: {INPUT_PATH}")

with INPUT_PATH.open("r", encoding="utf-8") as file:
    first_review = json.loads(next(file))

print(first_review)
print(first_review.keys())
```

Kết quả phải có tối thiểu ba field bắt buộc: `rating`, `title`, `text`.

### Bước 2 — đếm số record mà không nạp toàn bộ file vào RAM

```python
with INPUT_PATH.open("r", encoding="utf-8") as file:
    total_records = sum(1 for line in file if line.strip())

print(f"Tổng số record thô: {total_records:,}")
```

Không dùng `pandas.read_json()` để nạp toàn bộ file nếu máy thiếu RAM. Cách đọc từng dòng ở trên phù hợp với file hiện tại và sẽ được dùng cho bước làm sạch.

## 2.4. Schema User Reviews và các trường cần dùng

Theo tài liệu chính thức, một review thường có dạng:

```json
{
  "rating": 5.0,
  "title": "Such a lovely scent but not overpowering.",
  "text": "This spray is really nice. It smells really good...",
  "images": [],
  "asin": "B00YQ6X8EO",
  "parent_asin": "B00YQ6X8EO",
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "timestamp": 1588687728923,
  "helpful_vote": 0,
  "verified_purchase": true
}
```

| Field | Kiểu dữ liệu | Ý nghĩa | Dùng trong project? |
|---|---|---|---|
| `rating` | `float` | Điểm người dùng đánh giá, từ 1.0 đến 5.0 | **Có** — tạo label |
| `title` | `str` | Tiêu đề review | **Có** — ghép vào input |
| `text` | `str` | Nội dung review | **Có** — ghép vào input |
| `user_id` | `str` | ID người review | Tùy chọn — kiểm tra duplicate/phân tích |
| `asin` | `str` | ID của biến thể sản phẩm cụ thể | Tùy chọn |
| `parent_asin` | `str` | ID sản phẩm cha | Tùy chọn — dùng nếu nối metadata |
| `timestamp` | `int` | Unix timestamp, thường ở milliseconds | Tùy chọn — phân tích theo thời gian |
| `verified_purchase` | `bool` | Review có xác thực mua hàng hay không | Không dùng trong baseline |
| `helpful_vote` | `int` | Số lượt đánh giá hữu ích | Không dùng trong baseline |
| `images` | `list` | Ảnh do user đăng | Không dùng trong text-only model |

`asin` và `parent_asin` không giống nhau. Một `parent_asin` có thể đại diện cho nhiều biến thể màu, kích cỡ hoặc kiểu dáng; nếu sau này nối review với Item Metadata, dùng `parent_asin` làm khóa liên kết.

## 2.5. Kiểm tra dữ liệu ngay sau khi tải

Chạy kiểm tra trên 10.000 record đầu tiên trước khi làm sạch toàn bộ file. Mục đích là phát hiện file tải sai, file rỗng, schema thay đổi hoặc kiểu dữ liệu bất thường.

```python
import json
from collections import Counter
from pathlib import Path

INPUT_PATH = Path(r"All_Beauty.jsonl\All_Beauty.jsonl")
required = {"rating", "title", "text"}
rating_counts = Counter()
title_empty = 0
text_empty = 0
sample_size = 10_000

with INPUT_PATH.open("r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index == sample_size:
            break
        review = json.loads(line)
        missing = required - set(review)
        assert not missing, f"Thiếu field bắt buộc: {missing}"

        rating_counts[review.get("rating")] += 1
        title_empty += not bool(str(review.get("title") or "").strip())
        text_empty += not bool(str(review.get("text") or "").strip())

print("Số record mẫu:", sum(rating_counts.values()))
print("Phân bố rating:", dict(sorted(rating_counts.items(), key=lambda pair: str(pair[0]))))
print("Title rỗng:", title_empty)
print("Text rỗng:", text_empty)
```

Đây chỉ là kiểm tra mẫu. Số liệu chính thức trong report phải lấy từ `preprocessing_stats.json` sau khi xử lý toàn bộ file ở bước tiếp theo.

## 2.6. Chạy pipeline làm sạch và tạo nhãn

Tạo file `src/preprocess_all_beauty.py`, sao chép toàn bộ code dưới đây vào file đó, sau đó chạy từ thư mục project:

```python
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

# File input đã tải và giải nén.
INPUT_PATH = Path(r"All_Beauty.jsonl\All_Beauty.jsonl")

# Output được tạo tự động. Không ghi đè file raw.
OUTPUT_DIR = Path(r"data\interim")
CLEANED_PATH = OUTPUT_DIR / "cleaned.jsonl"
STATS_PATH = OUTPUT_DIR / "preprocessing_stats.json"


def normalize_for_check(value):
    """Đưa giá trị text về chuỗi dùng để kiểm tra rỗng và ghép review_text."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def label_from_rating(rating):
    if rating in (1.0, 2.0):
        return "Negative"
    if rating == 3.0:
        return "Neutral"
    if rating in (4.0, 5.0):
        return "Positive"
    return None


def exact_record_hash(record):
    """Hash của toàn bộ JSON record để chỉ loại các record trùng hoàn toàn."""
    canonical_record = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical_record.encode("utf-8")).digest()


if not INPUT_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stats = Counter()
label_counts = Counter()
seen_records = set()

with INPUT_PATH.open("r", encoding="utf-8") as source, \
     CLEANED_PATH.open("w", encoding="utf-8") as target:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            stats["blank_lines_removed"] += 1
            continue

        stats["raw_records"] += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid_json_removed"] += 1
            continue

        try:
            rating = float(record.get("rating"))
        except (TypeError, ValueError):
            stats["invalid_rating_removed"] += 1
            continue

        # Chỉ nhận năm mức rating để ánh xạ nhãn không mơ hồ.
        label = label_from_rating(rating)
        if label is None:
            stats["invalid_rating_removed"] += 1
            continue

        title = normalize_for_check(record.get("title"))
        text = normalize_for_check(record.get("text"))
        if not title and not text:
            stats["empty_title_and_text_removed"] += 1
            continue

        record_hash = exact_record_hash(record)
        if record_hash in seen_records:
            stats["duplicate_records_removed"] += 1
            continue
        seen_records.add(record_hash)

        # Giữ toàn bộ field gốc; chỉ thêm hai field phục vụ mô hình.
        cleaned_record = dict(record)
        cleaned_record["review_text"] = " ".join(part for part in (title, text) if part)
        cleaned_record["sentiment_class"] = label

        target.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
        stats["cleaned_records"] += 1
        label_counts[label] += 1

summary = {
    "input_file": str(INPUT_PATH),
    "output_file": str(CLEANED_PATH),
    "deduplication_rule": "Only exact duplicate JSON records are removed.",
    "processing_counts": dict(stats),
    "label_counts": dict(label_counts),
}
STATS_PATH.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
```

Chạy script:

```powershell
python src\preprocess_all_beauty.py
```

Sau khi chạy, kiểm tra ba giá trị phải có trong `preprocessing_stats.json`:

| Giá trị | Ý nghĩa trong report |
|---|---|
| `raw_records` | Tổng số record JSON đã đọc từ file nguồn |
| `invalid_rating_removed`, `empty_title_and_text_removed`, `duplicate_records_removed` | Số record bị loại theo từng lý do |
| `cleaned_records` | Số record còn lại trong `cleaned.jsonl` |
| `label_counts` | Số record Negative, Neutral, Positive |

Quy tắc loại được áp dụng theo thứ tự: rating → title/text → duplicate. Vì vậy, mỗi record bị loại chỉ được tính vào **một** lý do, và tổng các record bị loại cộng `cleaned_records` bằng `raw_records`.

## 2.7. Xuất ARFF từ `cleaned.jsonl`

Tạo file `src/export_arff.py` với code sau. File ARFF chỉ chứa hai attribute cần thiết cho Weka: `review_text` và `sentiment_class`.

```python
import json
from pathlib import Path

CLEANED_PATH = Path(r"data\interim\cleaned.jsonl")
ARFF_PATH = Path(r"data\processed\amazon_sentiment.arff")


def escape_arff(value):
    """Đưa text vào chuỗi quoted hợp lệ trong ARFF."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )


if not CLEANED_PATH.is_file():
    raise FileNotFoundError(f"Chưa có cleaned.jsonl: {CLEANED_PATH}")

ARFF_PATH.parent.mkdir(parents=True, exist_ok=True)

with CLEANED_PATH.open("r", encoding="utf-8") as source, \
     ARFF_PATH.open("w", encoding="utf-8", newline="\n") as target:
    target.write("@RELATION amazon_reviews_sentiment\n\n")
    target.write("@ATTRIBUTE review_text STRING\n")
    target.write("@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}\n\n")
    target.write("@DATA\n")

    for line in source:
        record = json.loads(line)
        text = escape_arff(record["review_text"])
        label = record["sentiment_class"]
        target.write(f"'{text}',{label}\n")

print(f"Đã tạo: {ARFF_PATH}")
```

Chạy export:

```powershell
python src\export_arff.py
```

Mỗi record trong `cleaned.jsonl` có giữ các field ban đầu và được thêm hai field sau:

```json
{
  "rating": 1.0,
  "title": "Battery died",
  "text": "It stopped working after two days.",
  "review_text": "Battery died It stopped working after two days.",
  "sentiment_class": "Negative"
}
```

## 2.8. Lưu ý riêng cho sentiment classification

1. **Dùng raw review file.** Đây là file duy nhất có đủ `rating`, `title` và `text`.
2. **Không cần Item Metadata trong baseline.** Metadata như giá, mô tả, thương hiệu mô tả sản phẩm, không phải cảm xúc của người review. Chỉ bổ sung nó khi có câu hỏi nghiên cứu mới.
3. **Không cần split có sẵn của dataset.** Các split `timestamp_split` được tạo cho bài toán recommendation và không có review text. Với sentiment classification, tự tạo train/validation/test split từ `cleaned.jsonl` bằng stratified split.
4. **Không dùng `rating` làm feature.** `rating` chỉ dùng để tạo `sentiment_class`; nếu đưa rating vào input, model sẽ biết trước đáp án và xảy ra data leakage.
5. **Nội dung review chủ yếu bằng tiếng Anh.** Dùng English stop words nếu áp dụng stop-word removal, nhưng giữ các từ phủ định như `not`, `no`, `never`.
6. **Cân nhắc tài nguyên.** Hãy hoàn thành pipeline trên 10.000–50.000 reviews trước, sau đó mới chạy toàn bộ category đã chọn.

## 2.9. Cấu trúc thư mục của project hiện tại

```text
 datamining\
├── All_Beauty.jsonl/
│   └── All_Beauty.jsonl              # File raw User Reviews đang dùng; không chỉnh sửa
├── data/
│   ├── interim/
│   │   └── cleaned.jsonl             # Sau lọc và tạo nhãn
│   └── processed/
│       └── amazon_sentiment.arff     # Dữ liệu cho Weka
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   └── metrics/
└── readme.md
```

Nguyên tắc: không chỉnh sửa `All_Beauty.jsonl\All_Beauty.jsonl`. Nếu có lỗi trong bước làm sạch, chạy lại script từ file raw này.

## 2.10. Trích dẫn dataset

Trong report, phần dữ liệu nên ghi rõ category đã dùng, thời điểm tải và link dataset. Ví dụ:

> Nghiên cứu sử dụng tập **User Reviews** của category **All_Beauty** thuộc Amazon Reviews 2023 do McAuley Lab công bố. Dataset được sử dụng để xây dựng bài toán phân loại cảm xúc ba lớp; nhãn được suy ra từ trường `rating`, còn đầu vào văn bản được ghép từ `title` và `text`.

BibTeX do trang dataset công bố:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

Các thông tin về schema, categories và ví dụ tải dữ liệu trong phần này được dựa trên [tài liệu chính thức của Amazon Reviews'23](https://amazon-reviews-2023.github.io/), [hướng dẫn đọc JSONL](https://amazon-reviews-2023.github.io/data_loading/jsonl.html) và [hướng dẫn dùng Hugging Face](https://amazon-reviews-2023.github.io/data_loading/huggingface.html).

---

# 3. Kiến thức nền tảng Python và xử lý dữ liệu

## 3.1. Python cơ bản cần học

Bạn không cần học toàn bộ Python trước khi làm project. Chỉ cần nắm các phần sau.

### Biến và kiểu dữ liệu

Python có các kiểu dữ liệu thường dùng:

```python
rating = 5
title = "Great product"
is_valid = True
price = 12.5
```

Các kiểu dữ liệu tương ứng:

| Kiểu | Ví dụ | Dùng trong project |
|---|---|---|
| `int` | `5` | rating, số lượng review |
| `float` | `4.5` | thống kê trung bình |
| `str` | `"Great product"` | title, text, label |
| `bool` | `True` | kiểm tra điều kiện |
| `list` | `["good", "cheap"]` | danh sách từ |
| `dict` | `{"rating": 5}` | một record JSON |

Một review ở dạng dictionary có thể trông như sau:

```python
review = {
    "user_id": "A123",
    "rating": 5,
    "title": "Great product",
    "text": "I really like this item."
}
```

### Điều kiện

Điều kiện được dùng để kiểm tra rating hợp lệ và tạo label.

```python
if rating >= 1 and rating <= 5:
    print("Rating hợp lệ")
else:
    print("Rating không hợp lệ")
```

Tạo nhãn sentiment:

```python
if rating in [1, 2]:
    label = "Negative"
elif rating == 3:
    label = "Neutral"
else:
    label = "Positive"
```

### Vòng lặp

Dữ liệu có thể gồm hàng nghìn hoặc hàng triệu review. Bạn cần duyệt qua từng record:

```python
for review in reviews:
    print(review["rating"])
```

Khi dùng `pandas`, thông thường bạn sẽ hạn chế vòng lặp từng dòng nếu có thể, vì xử lý theo cột thường nhanh hơn.

### Hàm

Hàm giúp chia code thành các phần rõ ràng.

```python
def create_sentiment_label(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"
```

Sau đó:

```python
label = create_sentiment_label(5)
```

Kết quả:

```text
Positive
```

---

## 3.2. JSON, JSONL, CSV và ARFF

### JSON

JSON là định dạng dữ liệu dạng key-value.

```json
{
  "user_id": "A123",
  "rating": 5,
  "title": "Amazing",
  "text": "Very good quality."
}
```

### JSONL

JSONL là viết tắt của JSON Lines. Mỗi dòng trong file là một JSON object độc lập.

Ví dụ file `reviews.jsonl`:

```json
{"user_id":"A1","rating":5,"title":"Amazing","text":"Very good quality."}
{"user_id":"A2","rating":1,"title":"Bad","text":"Stopped working."}
{"user_id":"A3","rating":3,"title":"","text":"It is okay."}
```

Điểm mạnh của JSONL là phù hợp với dữ liệu lớn vì có thể đọc từng dòng mà không cần tải toàn bộ file vào bộ nhớ.

Đọc JSONL bằng Pandas:

```python
import pandas as pd

df = pd.read_json("reviews.jsonl", lines=True)
```

### CSV

CSV là dạng bảng dữ liệu. Mỗi hàng là một record, mỗi cột là một attribute.

```csv
rating,title,text,sentiment_class
5,Amazing,Very good quality.,Positive
1,Bad,Stopped working.,Negative
```

Xuất CSV:

```python
df.to_csv("cleaned_reviews.csv", index=False)
```

### ARFF

ARFF là định dạng phổ biến trong Weka, một công cụ hỗ trợ data mining.

```arff
@RELATION amazon_reviews_sentiment

@ATTRIBUTE review_text STRING
@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}

@DATA
'Very good quality.',Positive
'Stopped working after one day.',Negative
'It is okay.',Neutral
```

Trong project này, ARFF có thể là file đầu vào cho Weka. Tuy nhiên, `JSONL` hoặc `CSV` vẫn nên được giữ lại làm dữ liệu trung gian vì dễ kiểm tra và tái sử dụng hơn.

---

## 3.3. Các thư viện Python cần thiết

### Pandas

Pandas dùng để xử lý dữ liệu dạng bảng.

```python
import pandas as pd
```

Các việc Pandas hỗ trợ:

- Đọc JSONL, CSV.
- Xem số dòng và số cột.
- Kiểm tra dữ liệu thiếu.
- Lọc record.
- Xóa record trùng.
- Tạo cột mới.
- Thống kê rating và label.
- Xuất dữ liệu đã làm sạch.

Ví dụ:

```python
df.head()
df.shape
df.columns
df["rating"].value_counts()
```

### NumPy

NumPy hỗ trợ tính toán số học và mảng số.

```python
import numpy as np
```

Trong project cơ bản, Pandas và scikit-learn đã sử dụng NumPy ở bên trong. Bạn chỉ cần biết NumPy ở mức cơ bản.

### Matplotlib và Seaborn

Hai thư viện này hỗ trợ trực quan hóa dữ liệu.

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Ví dụ biểu đồ phân bố rating:

```python
sns.countplot(data=df, x="rating")
plt.title("Rating distribution")
plt.show()
```

### Scikit-learn

Đây là thư viện quan trọng nhất cho phần machine learning.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

Scikit-learn sẽ được dùng để:

- Chia train/test.
- Chuyển text thành vector TF-IDF.
- Huấn luyện mô hình.
- Dự đoán class.
- Đánh giá Accuracy, Precision, Recall, F1-score.
- Tạo confusion matrix.

### NLTK hoặc spaCy

Dùng cho xử lý ngôn ngữ tự nhiên.

Các tác vụ thường gặp:

- Tokenization: tách câu thành từ.
- Stop words removal: bỏ từ quá phổ biến.
- Stemming: cắt từ về gốc đơn giản.
- Lemmatization: đưa từ về dạng từ điển.

Ví dụ:

```text
liked, likes, liking → like
```

Với project đầu tiên, bạn không cần dùng toàn bộ NLTK hoặc spaCy. TF-IDF với bước làm sạch nhẹ đã có thể là baseline tốt.

---

# 4. Khảo sát dữ liệu — Exploratory Data Analysis (EDA)

## 4.1. EDA là gì?

EDA là bước tìm hiểu dữ liệu trước khi xây dựng mô hình.

Mục tiêu không phải chỉ để vẽ biểu đồ đẹp. EDA giúp bạn phát hiện:

- Dữ liệu có bị thiếu không.
- Rating có hợp lệ không.
- Có bao nhiêu record trùng.
- Các lớp có bị mất cân bằng không.
- Review quá ngắn hoặc quá dài có nhiều không.
- Có thuộc tính nào không cần dùng không.

Nếu không khảo sát dữ liệu mà train mô hình ngay, bạn rất dễ có kết quả sai hoặc khó giải thích.

---

## 4.2. Các câu hỏi cần trả lời

### Có bao nhiêu record?

```python
total_records = len(df)
print(total_records)
```

Hoặc:

```python
df.shape
```

Nếu kết quả là:

```text
(100000, 10)
```

nghĩa là dữ liệu có:

- 100.000 review
- 10 attribute

### Dữ liệu có những attribute nào?

```python
print(df.columns.tolist())
```

Ví dụ kết quả:

```text
['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp']
```

Trong project này, các trường quan trọng nhất là:

| Attribute | Vai trò |
|---|---|
| `rating` | Tạo nhãn sentiment |
| `title` | Một phần nội dung review |
| `text` | Một phần nội dung review |
| `user_id` | Có thể dùng để phân tích thêm |
| `parent_asin` | ID sản phẩm, không bắt buộc dùng |
| `timestamp` | Có thể dùng cho phân tích theo thời gian |

### Có bao nhiêu review thiếu title hoặc text?

```python
df["title"].isna().sum()
df["text"].isna().sum()
```

Cần phân biệt:

- Giá trị `NaN` hoặc `None`.
- Chuỗi rỗng `""`.
- Chuỗi chỉ có khoảng trắng như `"   "`.

Ví dụ hàm kiểm tra một chuỗi rỗng:

```python
def is_empty_text(value):
    return pd.isna(value) or str(value).strip() == ""
```

### Phân bố rating như thế nào?

```python
df["rating"].value_counts().sort_index()
```

Kết quả ví dụ:

| Rating | Số review |
|---:|---:|
| 1 | 5.000 |
| 2 | 3.500 |
| 3 | 6.000 |
| 4 | 15.000 |
| 5 | 70.500 |

Từ bảng này, có thể thấy dữ liệu có khả năng mất cân bằng vì rating 5 sao rất nhiều.

### Review dài trung bình bao nhiêu từ?

Độ dài review có thể được tính theo số từ:

```python
df["review_length_words"] = df["review_text"].str.split().str.len()
average_length = df["review_length_words"].mean()
```

Bạn cũng nên báo cáo thêm:

- Trung vị — median.
- Độ dài nhỏ nhất.
- Độ dài lớn nhất.
- Các percentiles như 25%, 50%, 75%, 95%.

Ví dụ:

```python
df["review_length_words"].describe()
```

Điều này giúp bạn biết có bao nhiêu review rất ngắn như “Good”, “Bad”, hoặc rất dài như review vài nghìn từ.

---

## 4.3. Các biểu đồ nên đưa vào report

Một report tốt có thể gồm các biểu đồ sau:

1. Biểu đồ cột số lượng review theo rating 1–5.
2. Biểu đồ cột số lượng review theo sentiment class.
3. Histogram độ dài review.
4. Boxplot độ dài review theo từng sentiment class.
5. Confusion matrix của mô hình tốt nhất.

Ví dụ diễn giải:

> Dữ liệu có số lượng review 5 sao cao vượt trội, trong khi review 2 sao và 3 sao ít hơn. Sau khi ánh xạ rating thành sentiment class, lớp Positive chiếm tỷ lệ lớn nhất. Do đó, Accuracy không đủ để phản ánh toàn diện chất lượng mô hình; Macro F1-score được sử dụng làm chỉ số đánh giá chính.

---

# 5. Làm sạch dữ liệu

## 5.1. Mục tiêu làm sạch

Làm sạch dữ liệu nhằm bảo đảm mỗi record:

- Có rating hợp lệ.
- Có ít nhất title hoặc text.
- Không phải record trùng.
- Có nội dung đủ để sử dụng làm đầu vào mô hình.

Dữ liệu sau khi làm sạch sẽ được lưu thành:

```text
cleaned.jsonl
```

---

## 5.2. Quy tắc lọc dữ liệu

### Quy tắc 1: Chỉ giữ rating hợp lệ từ 1 đến 5

Giữ:

```text
1, 2, 3, 4, 5
```

Loại:

```text
0, 6, -1, null, NaN, "five"
```

Ví dụ:

```python
df = df[df["rating"].isin([1, 2, 3, 4, 5])]
```

### Quy tắc 2: Giữ review có title hoặc text

Các trường hợp:

| Title | Text | Xử lý |
|---|---|---|
| Có nội dung | Có nội dung | Giữ |
| Rỗng | Có nội dung | Giữ |
| Có nội dung | Rỗng | Giữ |
| Rỗng | Rỗng | Loại |

Ví dụ:

```python
title_not_empty = df["title"].fillna("").str.strip().ne("")
text_not_empty = df["text"].fillna("").str.strip().ne("")

df = df[title_not_empty | text_not_empty]
```

### Quy tắc 3: Loại record trùng

Có hai cách hiểu record trùng.

#### Cách 1: Trùng toàn bộ dòng

```python
df = df.drop_duplicates()
```

Phù hợp nếu dataset có những record hoàn toàn giống nhau.

#### Cách 2: Trùng các trường quan trọng

```python
df = df.drop_duplicates(
    subset=["user_id", "parent_asin", "rating", "title", "text"]
)
```

Cách này phù hợp hơn với project vì nó xem hai record là trùng nếu cùng người dùng, cùng sản phẩm, cùng rating và cùng nội dung review.

Trong report, bạn cần ghi rõ cách chọn. Ví dụ:

> Một record được xem là trùng nếu có cùng `user_id`, `parent_asin`, `rating`, `title` và `text`. Trong mỗi nhóm trùng, chỉ giữ lại record đầu tiên.

---

## 5.3. Ghi log số record bị loại

Bạn nên lưu các con số sau:

| Bước | Số record bị loại |
|---|---:|
| Tổng số record ban đầu | … |
| Rating không hợp lệ | … |
| Title và text cùng rỗng | … |
| Record trùng | … |
| Record còn lại | … |

Ví dụ diễn giải trong report:

> Dataset ban đầu gồm 100.000 records. Có 80 records bị loại vì rating không thuộc khoảng từ 1 đến 5. Tiếp theo, 1.200 records bị loại vì cả title và text đều không có nội dung. Sau khi loại 350 records trùng, dataset cuối cùng gồm 98.370 reviews.

Không nên viết số liệu minh họa này vào report cuối cùng; cần thay bằng số liệu thật sau khi chạy xử lý dữ liệu.

---

# 6. Tạo review_text và sentiment_class

## 6.1. Tạo review_text

Mục tiêu là hợp nhất title và text thành một đầu vào duy nhất.

```python
df["title"] = df["title"].fillna("").str.strip()
df["text"] = df["text"].fillna("").str.strip()

df["review_text"] = (
    df["title"] + " " + df["text"]
).str.replace(r"\s+", " ", regex=True).str.strip()
```

Ví dụ:

| Title | Text | Review text |
|---|---|---|
| Battery died | It stopped working after two days. | Battery died It stopped working after two days. |
| Rỗng | Very good quality. | Very good quality. |
| Good product | Rỗng | Good product |

Không nên để chuỗi `"nan"` xuất hiện trong `review_text`.

Sai:

```text
nan Very good quality.
```

Đúng:

```text
Very good quality.
```

---

## 6.2. Tạo sentiment_class

```python
def map_rating_to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    return None
```

Áp dụng:

```python
df["sentiment_class"] = df["rating"].apply(map_rating_to_sentiment)
```

Sau đó kiểm tra:

```python
df["sentiment_class"].value_counts()
```

---

## 6.3. Weak labeling và giới hạn của nhãn

Nhãn sentiment được suy ra từ rating, không phải do con người đọc từng review rồi gán nhãn. Đây được gọi là weak labeling.

Ví dụ mâu thuẫn có thể xảy ra:

| Rating | Nội dung review | Nhãn được tạo |
|---:|---|---|
| 5 | “Great product, but shipping was terrible.” | Positive |
| 1 | “Product is good but I received the wrong color.” | Negative |
| 3 | “Works fine, nothing impressive.” | Neutral |
| 3 | “Very good, but slightly expensive.” | Neutral |

Vì vậy, mô hình thực chất học cách dự đoán sentiment theo rating do người dùng chấm, không phải lúc nào cũng phản ánh hoàn toàn cảm xúc ngôn ngữ trong câu.

Trong report, có thể nêu:

> Sentiment label được tạo tự động từ rating. Phương pháp này hiệu quả cho dữ liệu lớn nhưng có thể tạo ra một số trường hợp nhiễu nhãn, khi nội dung review và rating không hoàn toàn nhất quán.

---

# 7. Tiền xử lý văn bản

## 7.1. Vì sao phải xử lý văn bản?

Máy tính không hiểu trực tiếp câu:

```text
"This product is absolutely amazing!"
```

Mô hình chỉ làm việc với số. Trước khi chuyển text thành số, cần chuẩn hóa một số phần không cần thiết.

---

## 7.2. Các bước làm sạch cơ bản

### Đưa về chữ thường

```text
Great Product → great product
```

Lợi ích: tránh coi `Good`, `good`, `GOOD` là ba từ khác nhau.

### Xóa khoảng trắng thừa

```text
"great    product" → "great product"
```

### Xóa URL

```text
"See https://example.com for details"
→ "See for details"
```

### Xóa HTML

```text
"<br>This is great!</br>"
→ "This is great!"
```

### Xử lý dấu câu

Có hai hướng:

- Xóa hoàn toàn dấu câu.
- Giữ lại một số dấu như `!` hoặc `?`.

Ví dụ:

```text
"Excellent!!!"
```

Dấu chấm than có thể thể hiện cảm xúc mạnh. Với project cơ bản, bạn có thể để `TfidfVectorizer` tự xử lý tokenization mặc định.

---

## 7.3. Tokenization

Tokenization là tách câu thành các đơn vị nhỏ hơn, thường là từ.

```text
"This product is very good"
```

Sau tokenization:

```text
["this", "product", "is", "very", "good"]
```

Trong scikit-learn, `TfidfVectorizer` đã có cơ chế tokenization mặc định. Vì thế bạn không nhất thiết phải tự tokenization nếu làm baseline.

---

## 7.4. Stop words

Stop words là những từ rất phổ biến nhưng thường ít mang ý nghĩa phân loại:

```text
the, is, are, a, an, and, of, to
```

Ví dụ:

```text
"This is a good product"
```

Sau khi loại stop words:

```text
"good product"
```

Tuy nhiên, không nên loại stop words một cách máy móc. Các từ như `not`, `no`, `never` rất quan trọng cho sentiment.

Ví dụ:

```text
"not good"
```

Nếu loại từ `not`, câu chỉ còn:

```text
"good"
```

và ý nghĩa bị đảo ngược.

Do đó, nếu dùng stop words, hãy kiểm tra danh sách stop words để không xóa các từ phủ định quan trọng.

---

## 7.5. Stemming và lemmatization

### Stemming

Stemming là cắt từ về dạng gốc đơn giản.

```text
liked → like
liking → like
studies → studi
```

Nhược điểm: kết quả có thể không phải từ đúng ngữ pháp.

### Lemmatization

Lemmatization đưa từ về dạng từ điển.

```text
liked → like
better → good
mice → mouse
```

Lemmatization thường có chất lượng ngôn ngữ tốt hơn nhưng cần xử lý phức tạp hơn.

Với project đầu tiên, bạn có thể bắt đầu với:

- Lowercase.
- Xóa URL.
- Xóa khoảng trắng thừa.
- Không dùng stemming/lemmatization ở baseline đầu tiên.

Sau đó tạo một phiên bản thử nghiệm có stop words hoặc lemmatization để so sánh.

---

## 7.6. Hai chiến lược preprocessing nên thử

### Chiến lược A: Làm sạch nhẹ

- Lowercase.
- Xóa URL.
- Xóa HTML.
- Chuẩn hóa khoảng trắng.
- Giữ hầu hết từ và dấu câu.

Ưu điểm:

- Đơn giản.
- Ít nguy cơ mất tín hiệu cảm xúc.
- Phù hợp làm baseline.

### Chiến lược B: Làm sạch nâng cao

- Lowercase.
- Xóa URL và HTML.
- Tokenization.
- Loại stop words có chọn lọc.
- Lemmatization.
- Sử dụng unigram và bigram.

Ưu điểm:

- Giảm vocabulary không cần thiết.
- Có thể cải thiện mô hình trong một số dataset.

Nhược điểm:

- Tốn thời gian.
- Có nguy cơ làm mất ngữ cảnh, đặc biệt với từ phủ định.

---

# 8. Biến đổi text thành đặc trưng số

## 8.1. Bag of Words

Bag of Words biểu diễn một review bằng số lần xuất hiện của từ.

Giả sử có ba review:

```text
1. great product
2. bad product
3. great quality
```

Vocabulary:

```text
["bad", "great", "product", "quality"]
```

Các vector:

| Review | bad | great | product | quality |
|---|---:|---:|---:|---:|
| great product | 0 | 1 | 1 | 0 |
| bad product | 1 | 0 | 1 | 0 |
| great quality | 0 | 1 | 0 | 1 |

Nhược điểm:

- Từ phổ biến như `product` có thể xuất hiện nhiều nhưng ít hữu ích.
- Không phản ánh rõ tầm quan trọng của từ.
- Không giữ thứ tự từ.

---

## 8.2. TF-IDF

TF-IDF là phương pháp nên dùng làm chính trong project.

TF-IDF bao gồm:

- TF — Term Frequency: từ xuất hiện bao nhiêu lần trong review.
- IDF — Inverse Document Frequency: từ đó có hiếm trong toàn bộ dataset không.

Công thức trực quan:

```text
TF-IDF = mức độ xuất hiện trong review × độ hiếm trong toàn dataset
```

Ví dụ:

- `product` xuất hiện trong hầu hết review → trọng số thấp hơn.
- `excellent`, `broken`, `refund`, `disappointed` xuất hiện ít hơn và mang ý nghĩa rõ → trọng số cao hơn.

Sử dụng với scikit-learn:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=20000
)
```

Các tham số cần hiểu:

| Tham số | Ý nghĩa |
|---|---|
| `lowercase=True` | Chuyển text về chữ thường |
| `ngram_range=(1, 1)` | Chỉ dùng từ đơn |
| `ngram_range=(1, 2)` | Dùng từ đơn và cụm 2 từ |
| `min_df=2` | Loại từ chỉ xuất hiện ở 1 review |
| `max_df=0.95` | Loại từ xuất hiện trong trên 95% review |
| `max_features=20000` | Giới hạn vocabulary tối đa 20.000 đặc trưng |

---

## 8.3. N-grams

N-gram là một cụm gồm `n` từ liên tiếp.

| Loại | Ví dụ với “this product is very good” |
|---|---|
| Unigram | this, product, is, very, good |
| Bigram | this product, product is, is very, very good |
| Trigram | this product is, product is very, is very good |

Bigram rất quan trọng trong sentiment analysis.

Ví dụ:

```text
not good
very bad
highly recommend
waste money
works great
```

Nếu chỉ dùng unigram, “not good” bị tách thành:

```text
not
good
```

Mô hình có thể bị ảnh hưởng bởi từ `good` tích cực. Bigram giúp giữ cụm `not good` như một đặc trưng tiêu cực.

Khuyến nghị cho project:

```python
ngram_range=(1, 2)
```

---

# 9. Chia dữ liệu train, validation và test

## 9.1. Vì sao không dùng toàn bộ dữ liệu để train?

Nếu train và đánh giá trên cùng một dữ liệu, mô hình có thể “học thuộc” dữ liệu thay vì học quy luật tổng quát.

Kết quả đánh giá sẽ cao một cách không trung thực.

Do đó cần tách dữ liệu thành các phần riêng.

| Tập dữ liệu | Mục đích |
|---|---|
| Training set | Dùng để mô hình học |
| Validation set | Chọn tham số, chọn mô hình |
| Test set | Đánh giá cuối cùng |

---

## 9.2. Hai cách chia phổ biến

### Cách 1: Train / Validation / Test

| Tập | Tỷ lệ |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Dùng validation để so sánh các cấu hình khác nhau.

### Cách 2: Train / Test + Cross-validation

| Tập | Tỷ lệ |
|---|---:|
| Train | 80% |
| Test | 20% |

Sau đó dùng cross-validation trên training set.

Cách này phù hợp với scikit-learn và thường được dùng trong project học thuật.

---

## 9.3. Stratified split

Nếu lớp Positive có 80% dữ liệu, lớp Neutral chỉ có 5%, việc chia ngẫu nhiên bình thường có thể khiến tập test có quá ít review Neutral.

Stratified split giữ tỷ lệ các lớp gần giống nhau giữa train và test.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"],
    df["sentiment_class"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment_class"]
)
```

`random_state=42` giúp bạn có thể chạy lại và nhận được cùng cách chia dữ liệu.

---

## 9.4. Data leakage

Data leakage là khi thông tin từ test set vô tình bị đưa vào quá trình huấn luyện.

Một lỗi phổ biến:

```python
# Sai
X_all = vectorizer.fit_transform(df["review_text"])
X_train, X_test, y_train, y_test = train_test_split(X_all, y)
```

Vấn đề: TF-IDF đã nhìn thấy toàn bộ vocabulary và IDF của test set.

Cách đúng:

```python
X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"],
    df["sentiment_class"],
    test_size=0.2,
    stratify=df["sentiment_class"],
    random_state=42
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

Nguyên tắc:

> Mọi bước “học từ dữ liệu”, bao gồm TF-IDF, scaling, oversampling và chọn feature, chỉ được fit trên training set.

---

# 10. Các mô hình nên thử

## 10.1. Dummy Classifier

Dummy Classifier là baseline đơn giản. Nó không thật sự học nội dung text.

Ví dụ, nó có thể luôn dự đoán lớp phổ biến nhất là `Positive`.

Nếu Dummy Classifier đã đạt Accuracy 75%, mà mô hình của bạn đạt 76%, mô hình của bạn chưa thực sự tốt.

Mục đích của Dummy Classifier là làm mốc so sánh tối thiểu.

---

## 10.2. Multinomial Naive Bayes

Naive Bayes là thuật toán cổ điển cho text classification.

Ý tưởng đơn giản:

- Mô hình xem xét xác suất một từ xuất hiện trong mỗi class.
- Các từ như `excellent`, `love`, `perfect` có thể xuất hiện nhiều ở Positive.
- Các từ như `broken`, `terrible`, `refund` có thể xuất hiện nhiều ở Negative.

Ưu điểm:

- Nhanh.
- Dễ triển khai.
- Hoạt động tốt với Bag of Words hoặc TF-IDF.
- Là baseline rất phổ biến.

Nhược điểm:

- Giả định các feature độc lập với nhau, điều này không hoàn toàn đúng với ngôn ngữ.
- Có thể không tốt bằng Linear SVM hoặc Logistic Regression.

---

## 10.3. Logistic Regression

Mặc dù có chữ “Regression”, Logistic Regression thường được sử dụng cho classification.

Mô hình học trọng số cho từng từ hoặc cụm từ.

Ví dụ:

| Feature | Xu hướng trọng số |
|---|---|
| excellent | Positive |
| highly recommend | Positive |
| broken | Negative |
| waste money | Negative |
| okay | Neutral |

Ưu điểm:

- Mạnh và ổn định với TF-IDF.
- Có thể dùng `class_weight="balanced"`.
- Có thể xem trọng số feature để giải thích mô hình.
- Có thể cung cấp xác suất dự đoán.

Khuyến nghị:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
```

---

## 10.4. Linear SVM

Linear SVM thường cho hiệu quả rất tốt trong text classification có số lượng feature lớn.

Ưu điểm:

- Hiệu quả cao trên dữ liệu sparse như TF-IDF.
- Thường cạnh tranh hoặc tốt hơn Logistic Regression.
- Phù hợp với dataset lớn.

Nhược điểm:

- `LinearSVC` không cung cấp xác suất dự đoán trực tiếp.
- Cần tinh chỉnh tham số `C`.

Khuyến nghị:

```python
from sklearn.svm import LinearSVC

model = LinearSVC(class_weight="balanced")
```

---

## 10.5. Thứ tự thử nghiệm

Nên thực hiện theo thứ tự:

1. Dummy Classifier.
2. TF-IDF + Multinomial Naive Bayes.
3. TF-IDF + Logistic Regression.
4. TF-IDF + Linear SVM.
5. So sánh kết quả.
6. Tinh chỉnh mô hình tốt nhất.

Bảng ghi kết quả thí nghiệm:

| Mô hình | TF-IDF | N-grams | Class weight | Accuracy | Macro F1 |
|---|---|---|---|---:|---:|
| Dummy | Không | Không | Không | … | … |
| Naive Bayes | Có | (1,1) | Không | … | … |
| Logistic Regression | Có | (1,2) | Balanced | … | … |
| Linear SVM | Có | (1,2) | Balanced | … | … |

---

# 11. Đánh giá mô hình

## 11.1. Accuracy

Accuracy là tỷ lệ dự đoán đúng trên toàn bộ dữ liệu.

```text
Accuracy = số dự đoán đúng / tổng số dự đoán
```

Ví dụ:

- Có 100 review.
- Mô hình đoán đúng 80 review.
- Accuracy = 80%.

Hạn chế: Accuracy gây hiểu lầm khi dữ liệu mất cân bằng.

---

## 11.2. Precision

Precision trả lời câu hỏi:

> Trong các review mà mô hình dự đoán là Positive, bao nhiêu review thực sự Positive?

```text
Precision = TP / (TP + FP)
```

Trong đó:

- TP: dự đoán Positive và thực tế Positive.
- FP: dự đoán Positive nhưng thực tế không Positive.

---

## 11.3. Recall

Recall trả lời câu hỏi:

> Trong tất cả review thực sự Positive, mô hình nhận diện được bao nhiêu review?

```text
Recall = TP / (TP + FN)
```

Trong đó:

- FN: thực tế Positive nhưng mô hình dự đoán lớp khác.

---

## 11.4. F1-score

F1-score là trung bình điều hòa của Precision và Recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

F1-score cao khi cả Precision và Recall đều cao.

---

## 11.5. Macro F1 và Weighted F1

### Macro F1

Tính F1 riêng cho từng class rồi lấy trung bình đều.

```text
Macro F1 = (F1 Negative + F1 Neutral + F1 Positive) / 3
```

Macro F1 coi ba lớp quan trọng như nhau, kể cả Neutral có ít mẫu.

### Weighted F1

Tính trung bình F1 theo số lượng mẫu của mỗi lớp.

Lớp lớn như Positive sẽ có ảnh hưởng nhiều hơn.

Khuyến nghị:

- Dùng Macro F1 làm chỉ số chính để chọn mô hình.
- Báo cáo thêm Accuracy và Weighted F1.

---

## 11.6. Confusion matrix

Confusion matrix cho biết mô hình đang nhầm class nào với class nào.

Ví dụ:

| Thực tế \ Dự đoán | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 800 | 120 | 80 |
| Neutral | 100 | 250 | 150 |
| Positive | 50 | 90 | 3360 |

Diễn giải:

- 800 review Negative được dự đoán đúng là Negative.
- 150 review Neutral bị đoán nhầm thành Positive.
- Lớp Neutral có thể là lớp khó dự đoán nhất.

Trong sentiment analysis, việc nhầm `Neutral` với `Positive` hoặc `Negative` là bình thường, vì review 3 sao thường không thể hiện cảm xúc rõ rệt.

---

# 12. Mất cân bằng lớp

## 12.1. Class imbalance là gì?

Class imbalance xảy ra khi số lượng mẫu giữa các class chênh lệch lớn.

Ví dụ:

| Class | Số lượng | Tỷ lệ |
|---|---:|---:|
| Negative | 10.000 | 10% |
| Neutral | 8.000 | 8% |
| Positive | 82.000 | 82% |

Nếu mô hình luôn đoán Positive, nó có Accuracy 82%, nhưng không có ích trong việc nhận diện review tiêu cực hoặc trung tính.

---

## 12.2. Cách xử lý

### Dùng class_weight="balanced"

Đây là lựa chọn nên thử đầu tiên.

```python
LogisticRegression(class_weight="balanced")
```

Hoặc:

```python
LinearSVC(class_weight="balanced")
```

Mô hình sẽ phạt nặng hơn khi dự đoán sai lớp hiếm.

### Undersampling

Giảm bớt số lượng mẫu của lớp lớn nhất.

Ví dụ giảm Positive từ 82.000 xuống 20.000.

Ưu điểm:

- Làm dữ liệu cân bằng hơn.
- Có thể train nhanh hơn.

Nhược điểm:

- Loại bỏ nhiều dữ liệu tốt.
- Có thể mất đa dạng trong lớp Positive.

### Oversampling

Sao chép hoặc tạo thêm sample cho lớp ít.

Nhược điểm với text:

- Copy nguyên review có thể gây overfitting.
- Chỉ nên dùng cẩn thận và thực hiện trên training set, không phải trên toàn bộ dataset.

### Báo cáo Macro F1

Dù có dùng kỹ thuật cân bằng hay không, Macro F1 vẫn cần có trong kết quả cuối cùng.

---

# 13. Xuất file cleaned.jsonl và ARFF

## 13.1. Nội dung cleaned.jsonl

File `cleaned.jsonl` nên giữ lại:

- Các attribute ban đầu cần thiết.
- `review_text`.
- `sentiment_class`.

Ví dụ:

```json
{
  "user_id": "A123",
  "parent_asin": "B001XYZ",
  "rating": 1,
  "title": "Battery died",
  "text": "It stopped working after two days.",
  "review_text": "Battery died It stopped working after two days.",
  "sentiment_class": "Negative"
}
```

Xuất bằng Pandas:

```python
df.to_json(
    "cleaned.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)
```

---

## 13.2. Nội dung file ARFF

Nếu chỉ dùng hai trường cho mô hình Weka:

```arff
@RELATION amazon_reviews_sentiment

@ATTRIBUTE review_text STRING
@ATTRIBUTE sentiment_class {Negative,Neutral,Positive}

@DATA
'Battery died It stopped working after two days.',Negative
'It is okay but nothing special.',Neutral
'Excellent product highly recommended.',Positive
```

Lưu ý khi xuất ARFF:

- Không để review có line break thật.
- Escape dấu nháy đơn.
- Bảo đảm class đúng chính tả và cùng định dạng.
- Không để giá trị thiếu trong `review_text`.
- Đặt `sentiment_class` là attribute cuối cùng.

Ngoài ARFF, nên lưu CSV để kiểm tra dễ hơn:

```python
df[["review_text", "sentiment_class"]].to_csv(
    "amazon_sentiment.csv",
    index=False
)
```

---

# 14. Quy trình thực hiện project

```text
1. Chọn một category từ Amazon Reviews 2023
                ↓
2. Tải file User Reviews JSONL
                ↓
3. Đọc dữ liệu bằng Pandas
                ↓
4. Khảo sát dữ liệu ban đầu
                ↓
5. Lọc rating không hợp lệ
                ↓
6. Loại review có title và text cùng rỗng
                ↓
7. Loại record trùng
                ↓
8. Tạo review_text
                ↓
9. Tạo sentiment_class từ rating
                ↓
10. Thống kê và trực quan hóa dữ liệu sau làm sạch
                ↓
11. Xuất cleaned.jsonl và ARFF
                ↓
12. Chia train/test theo stratified split
                ↓
13. Fit TF-IDF trên training set
                ↓
14. Train các mô hình
                ↓
15. Đánh giá bằng Accuracy, Macro F1, confusion matrix
                ↓
16. Phân tích lỗi và viết report
```

---

# 15. Đề cương report gợi ý

## Chương 1 — Giới thiệu

- Bối cảnh: đánh giá sản phẩm trực tuyến có số lượng rất lớn.
- Vấn đề: cần tự động xác định cảm xúc của người dùng.
- Mục tiêu: phân loại review thành Negative, Neutral, Positive.
- Dataset: Amazon Reviews 2023.
- Phạm vi: sử dụng một category và User Reviews.

## Chương 2 — Dữ liệu

- Mô tả các attribute.
- Số lượng record ban đầu.
- Các trường được sử dụng: `rating`, `title`, `text`.
- Quy tắc tạo nhãn sentiment.
- Phân bố rating và label.

## Chương 3 — Tiền xử lý dữ liệu

- Lọc rating hợp lệ.
- Xử lý title/text rỗng.
- Loại duplicate.
- Ghép `title + text`.
- Làm sạch văn bản.
- Xuất `cleaned.jsonl` và ARFF.
- Báo cáo số record bị loại ở từng bước.

## Chương 4 — Phương pháp

- Train/test split.
- TF-IDF.
- N-grams.
- Dummy Classifier.
- Naive Bayes.
- Logistic Regression.
- Linear SVM.
- Chỉ số đánh giá.

## Chương 5 — Kết quả thực nghiệm

- Bảng so sánh các mô hình.
- Accuracy, Precision, Recall, F1-score.
- Macro F1.
- Confusion matrix.
- Mô hình tốt nhất.
- Các tham số tốt nhất.

## Chương 6 — Phân tích và thảo luận

- Lớp nào khó dự đoán nhất.
- Các lỗi điển hình.
- Ảnh hưởng của class imbalance.
- Ảnh hưởng của bigram.
- Giới hạn của nhãn tạo từ rating.
- Hướng phát triển: BERT, aspect-based sentiment analysis, cân bằng dữ liệu tốt hơn.

## Chương 7 — Kết luận

- Tóm tắt quy trình.
- Mô hình tốt nhất.
- Kết quả chính.
- Hạn chế và hướng mở rộng.

---

# 16. Lộ trình học đề xuất

## Tuần 1: Python và Pandas

Học:

- Python cơ bản.
- List, dictionary, function.
- Pandas DataFrame.
- Đọc JSONL/CSV.
- Missing values.
- Filtering.
- Duplicate.
- Groupby và value_counts.

Mục tiêu:

> Có thể đọc file review, kiểm tra dữ liệu và tạo `cleaned.jsonl`.

## Tuần 2: Data preprocessing và EDA

Học:

- Làm sạch title/text.
- Tạo `review_text`.
- Tạo label.
- Thống kê mô tả.
- Vẽ biểu đồ.
- Phân bố dữ liệu.
- Class imbalance.

Mục tiêu:

> Hoàn thành dataset đã làm sạch, biểu đồ và bảng thống kê cho report.

## Tuần 3: Machine Learning cơ bản

Học:

- Supervised learning.
- Classification.
- Train/test split.
- Stratified split.
- Data leakage.
- Accuracy, Precision, Recall, F1-score.
- Confusion matrix.

Mục tiêu:

> Hiểu cách đánh giá mô hình một cách đúng đắn.

## Tuần 4: NLP và TF-IDF

Học:

- Text cleaning.
- Tokenization.
- Stop words.
- Bag of Words.
- TF-IDF.
- N-grams.

Mục tiêu:

> Biến `review_text` thành vector số để train mô hình.

## Tuần 5: Huấn luyện và so sánh mô hình

Học và thực hành:

- Dummy Classifier.
- Multinomial Naive Bayes.
- Logistic Regression.
- Linear SVM.
- `class_weight`.
- Hyperparameter tuning.

Mục tiêu:

> Chọn mô hình tốt nhất dựa trên Macro F1-score.

## Tuần 6: Hoàn thiện report

Hoàn thành:

- Bảng mô tả dữ liệu.
- Bảng làm sạch dữ liệu.
- Biểu đồ rating và label.
- Bảng kết quả các mô hình.
- Confusion matrix.
- Phân tích lỗi.
- Kết luận và hướng phát triển.

---

# 17. Checklist trước khi nộp project

## Dữ liệu

- [ ] Đã chọn một category rõ ràng.
- [ ] Đã nêu nguồn dữ liệu.
- [ ] Đã kiểm tra số record ban đầu.
- [ ] Đã loại rating không hợp lệ.
- [ ] Đã xử lý title/text rỗng.
- [ ] Đã loại record trùng.
- [ ] Đã ghi số record bị loại theo từng lý do.
- [ ] Đã tạo `review_text`.
- [ ] Đã tạo `sentiment_class`.
- [ ] Đã xuất `cleaned.jsonl`.
- [ ] Đã xuất ARFF.

## Mô hình

- [ ] Đã chia train/test bằng stratified split.
- [ ] Không có data leakage.
- [ ] Đã dùng TF-IDF.
- [ ] Đã thử ít nhất Naive Bayes, Logistic Regression và Linear SVM.
- [ ] Đã có baseline Dummy Classifier.
- [ ] Đã xử lý hoặc thảo luận class imbalance.
- [ ] Đã chọn mô hình tốt nhất dựa trên Macro F1.

## Đánh giá

- [ ] Có Accuracy.
- [ ] Có Precision, Recall, F1-score từng class.
- [ ] Có Macro F1 và Weighted F1.
- [ ] Có confusion matrix.
- [ ] Có phân tích các lỗi dự đoán.

## Report

- [ ] Mô tả dataset rõ ràng.
- [ ] Mô tả preprocessing rõ ràng.
- [ ] Báo cáo số liệu thực tế, không dùng số minh họa.
- [ ] Có bảng so sánh mô hình.
- [ ] Có biểu đồ trực quan.
- [ ] Nêu giới hạn của việc tạo label từ rating.
- [ ] Có kết luận và hướng phát triển.

# Kết luận

Đối với project đầu tiên, mục tiêu tốt nhất không phải là dùng mô hình phức tạp nhất. Mục tiêu là xây dựng một quy trình rõ ràng, có thể lặp lại và giải thích được:

```text
Dữ liệu thô
→ Làm sạch có thống kê
→ Tạo review_text và sentiment label
→ TF-IDF
→ So sánh nhiều mô hình
→ Đánh giá đúng chỉ số
→ Phân tích lỗi và giới hạn
```

Nếu hoàn thành được quy trình này bằng TF-IDF kết hợp Naive Bayes, Logistic Regression và Linear SVM, bạn đã có một project Data Mining/NLP đầy đủ và phù hợp để làm báo cáo hoặc đồ án.
