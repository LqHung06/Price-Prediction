# Price-Prediction

Dự đoán giá nhà và giá kim cương bằng hai mô hình hồi quy, Ridge Regression và Random Forest Regression, trên 4 bộ dữ liệu Kaggle. Mục tiêu từ đầu là R² trên tập test vượt 0,80. Cả 4 bộ chạy cùng một pipeline để so sánh được với nhau: target là log của giá, chia ngẫu nhiên 80/20 với `random_state=42`, impute median/mode chỉ học từ train, winsorise ở 1%/99%.

## Kết quả chính

Bảng dưới báo R² trên cả hai thang. Thang log là thang mô hình học, thang giá thật là thang người đọc quan tâm.

| Bộ dữ liệu | Dòng | Ridge R² log / thô | RF R² log / thô | MAE (Ridge / RF) | Kết luận |
|---|---|---|---|---|---|
| Nhà Ấn Độ | 1.124 | 0,9427 / 0,9126 | 0,8900 / 0,9009 | 25.938 / 36.073 | Cả hai đều qua mốc, Ridge thắng |
| Melbourne | 13.580 | 0,7936 / 0,7583 | 0,8685 / 0,8175 | 198.759 / 158.610 AUD | Chỉ RF qua mốc, Ridge trượt |
| King County | 21.613 | 0,8793 / 0,8366 | 0,8867 / 0,8545 | $79.091 / $74.654 | Cả hai đều qua mốc, RF nhỉnh hơn |
| Kim cương | 53.920 | 0,9834 / 0,9670 | 0,9882 / 0,9805 | $390 / $277 | Cả hai đều qua mốc, cao nhất |

Đơn vị tiền tệ khác nhau theo từng bộ, theo đúng đơn vị trong file gốc.

## Bài học từ vòng tối ưu tiền xử lý

Sau vòng chạy chuẩn, Melbourne là bộ duy nhất còn dư địa cải thiện. Bộ này thiếu nặng nhất, 47,5% diện tích nhà trống và 314 vùng ngoại ô, và Ridge trượt mốc 0,80. Mỗi phương án được thử riêng, so sánh chỉ bằng 5-fold CV trên train, rồi phương án thắng mới chạm test một lần.

Thêm tương tác (lat×long, Distance×Type, log_Landsize×Rooms) đưa Ridge lên R² log 0,8042 trên test. Gộp 3 phương án, tương tác cộng KNN impute cộng gộp vùng ngoại ô thưa, là tốt nhất: CV 0,8144, test R² log 0,8183 và R² thô 0,7902, MAE 184.621 AUD. Thang thô vẫn thiếu mốc 0,80 một chút. Vượt mốc đó cần thêm thứ khác ngoài tiền xử lý, ví dụ một họ mô hình khác hoặc biến mới.

Gộp 3 không phải lúc nào cũng thắng. Ở Nhà Ấn Độ, phương án thắng là pair2, tương tác cộng log_Area², với R² log 0,9444 trên test, còn gộp 3 chỉ đạt CV 0,9015, thua cả baseline. KNN impute làm hại ở đây vì chỉ 8,6% dòng thiếu và Area gần như không tương quan với Rooms. Ở King County, tương tác đơn thắng với R² log 0,8801 trên test, còn gộp 3 tệ nhất cho Ridge vì gộp zipcode thưa xóa đúng tín hiệu vị trí mà Ridge dựa vào.

Chi tiết đầy đủ nằm trong `reports/bao-cao-toi-uu-tien-xu-ly-melbourne.html` và `reports/bao-cao-gop-3-phuong-an-melbourne.html`.

## Các bộ dữ liệu

| Bộ | Nguồn Kaggle | File | Ghi chú |
|---|---|---|---|
| Nhà Ấn Độ | [raviiyadavv/house-price-predecition](https://www.kaggle.com/datasets/raviiyadavv/house-price-predecition) | `notebooks/dataset_2.csv` | File thật là `dataset_2.csv`, 1.124 dòng, khác với bộ Housing.csv 545 dòng mà mô tả Kaggle gợi tới |
| Melbourne | [dansbecker/melbourne-housing-snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) | `notebooks/melb_data.csv` | Bán nhà Melbourne 2016-2017, AUD; BuildingArea trống 47,5% |
| King County | [harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) | `notebooks/kc_house_data.csv` | Bán nhà King County (Seattle) 05/2014-05/2015, USD; 0 ô trống |
| Kim cương | [shivam2503/diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds) | `data/diamonds.csv` | 53.940 dòng, đã bỏ 20 dòng x/y/z = 0 |

## Cấu trúc repo

```
Price-Prediction/
├── README.md
├── requirements.txt
├── notebooks/                  # 3 notebook đã chạy xong, mở là xem được kết quả
│   ├── phan-tich-gia-nha-an-do.ipynb
│   ├── phan-tich-gia-nha-king-county.ipynb
│   ├── phan-tich-gia-nha-melbourne.ipynb
│   ├── dataset_2.csv           # để cạnh notebook để chạy ngay, không cần sửa đường dẫn
│   ├── kc_house_data.csv
│   └── melb_data.csv
├── data/
│   └── diamonds.csv            # chưa có notebook riêng, chạy bằng script trong src/
├── reports/                    # 6 báo cáo HTML, mở trực tiếp bằng trình duyệt
│   ├── links.txt               # link GitHub Pages của 6 báo cáo
│   ├── bao-cao-gia-nha-king-county.html
│   ├── bao-cao-gia-nha-melbourne.html
│   ├── bao-cao-gia-nha-an-do.html
│   ├── bao-cao-gia-kim-cuong.html
│   ├── bao-cao-toi-uu-tien-xu-ly-melbourne.html
│   └── bao-cao-gop-3-phuong-an-melbourne.html
├── src/                        # script chạy lại toàn bộ pipeline, đường dẫn tương đối
│   ├── eda_*.py                # phân tích khám phá
│   ├── model_*.py              # chạy Ridge + RF, ghi results.json + biểu đồ
│   └── preproc_opt/            # vòng tối ưu tiền xử lý, chia theo từng bộ
│       ├── melbourne/
│       ├── indian-house/
│       └── king-county/
└── results/                    # results.json + test_predictions.csv của từng bộ
    ├── indian-house/
    ├── melbourne/
    ├── king-county/
    └── diamonds/
```

Chạy lại script `model_*.py` sẽ ghi đè `results.json`/`test_predictions.csv` và sinh thêm thư mục `charts/` trong thư mục kết quả tương ứng.

## Cách chạy lại

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Cách 1: mở notebook trong notebooks/ bằng Jupyter hoặc Google Colab, chạy từ trên xuống.
# File CSV đã nằm cạnh notebook nên không cần sửa đường dẫn.

# Cách 2: chạy script, ví dụ bộ Nhà Ấn Độ (chạy từ thư mục src/)
cd src
python model_house.py        # ghi kết quả vào results/indian-house/
python eda_house.py          # chỉ in thống kê EDA ra màn hình
```

Môi trường đã kiểm chứng: Python 3.12, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2. `random_state=42` cố định ở mọi chỗ. Phiên bản thư viện khác có thể làm số lệch ở chữ số thập phân thứ 3-4.

Random Forest 300 cây trên King County và kim cương chạy khá lâu, mức chục phút. Vòng tối ưu Melbourne dùng 150 cây cho CV để tiết kiệm thời gian, nhưng các con số chính trên test vẫn tính bằng 300 cây.

## Phương pháp chung

- Target: log(giá). Giá nhà và giá kim cương đều lệch phải mạnh. Log làm phân phối gần chuẩn và quan hệ gần tuyến tính hơn.
- Chia tập: ngẫu nhiên 80/20, seed 42. Melbourne và King County có cột ngày, nhưng bài toán là định giá tại thời điểm giao dịch, nên không chia theo thời gian.
- Tiền xử lý: median/mode impute học từ train, winsorise biến số ở 1%/99% của train, OneHot cho biến phân loại. Không có cột rò rỉ target trong cả 4 bộ.
- Mô hình: Ridge với RidgeCV chọn alpha trong [0,1; 1; 10; 100; 1000]; Random Forest 300-500 cây, `min_samples_leaf=2`.
- Đánh giá: R² trên cả thang log và thang giá thật, MAE, RMSE, MedAPE, kèm 5-fold CV trên train để kiểm tra overfitting. Train và test R² gần nhau thì mô hình học thật, chênh lớn thì học vẹt.

## Lưu ý khi đọc kết quả

Kim cương R² ~0,99 không có nghĩa mô hình giỏi. Quan hệ carat và giá gần như một công thức, nên con số này là trần trên để đối chiếu. Bộ này cũng là bộ sạch nhất, 0 ô trống, nên gần như không có gì để tiền xử lý.

Melbourne là bộ tiền xử lý khó nhất: thiếu 47,5%/39,6%, Landsize lệch cực mạnh với 1.939 số 0, 314 vùng ngoại ô phải gộp. Đây cũng là bộ duy nhất mà gộp 3 phương án tiền xử lý thắng.

Permutation importance không tách được hai biến gần như trùng nhau. Ở kim cương, volume và carat tương quan 0,976 nên không kết luận được biến kích thước nào quan trọng hơn. Ở Melbourne, Rooms và Bedroom2 tương quan 0,94 nên Bedroom2 đã bị bỏ.

King County hợp với bài toán kinh tế nhất. Đây là ví dụ rõ của định giá hedonic: giá là hàm của diện tích, hạng nhà, vị trí, view và mặt nước, với câu chuyện vị trí rõ ràng. Medina trung bình 2,16 triệu USD, zip rẻ nhất 234 nghìn USD.

## Báo cáo

6 báo cáo HTML trong `reports/` viết bằng tiếng Việt, theo cùng một khung: bài toán, thu thập dữ liệu, làm sạch và chuẩn hóa, mô tả dữ liệu, lý thuyết hai mô hình, kết quả, kiểm tra overfitting, kết luận. Mở trực tiếp bằng trình duyệt, không cần server. Bật GitHub Pages cho repo thì xem được báo cáo bằng link thay vì chỉ xem mã nguồn.
