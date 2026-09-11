# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.13.15 / 2.11.0+cu128 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): class_id = 468,
class_name =  "cab",
rank = 1,
score = 0.510915,
taxonomy_name = "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào? 
Model nhận toàn bộ ảnh "traffic" làm input và cho rằng lớp "ImageNet-1K" phù hợp nhất với toàn ảnh là "cab", với score = 0.510915: Xét theo taxonomy ("ImageNet-1K") và nhiệm vụ classification mà model được huấn luyện, "cab" là lớp mà model đánh giá có độ tin cậy dự đoán cao nhất cho toàn ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? 
Checkpoint classification này được huấn luyện để dự đoán các lớp thuộc ImageNet-1K, các `class_id` là các index của class trong class space của model.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? 
ID = định dang có tính ổn định trong một taxonomy cụ thể, tên lớp giúp con người đọc. ID chỉ có ý nghĩa trong một taxomomy cụ thể 
class_id = machine-readable identity
class_name = human-readable meaning
taxonomy_name = semantic namespace
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? 
Task type: classification / multi-label classification / detection / segmentation.
Annotation unit: toàn ảnh hay từng object.
Multiple objects: cho phép nhiều class hay bắt buộc một class.
Primary object: nếu chỉ chọn một class thì chọn theo tiêu chí nào.
Occlusion: vật thể bị che một phần xử lý thế nào.
Ambiguity: không đủ thông tin để xác định class thì gán gì.
Ground truth source: lấy label từ dataset nào hoặc annotator nào.
- Vì sao model score không phải ground truth? 
Prediction là claim của model, score chỉ trả lời tương đối câu hỏi: Model tin prediction này đến mức nào.
Ground truth là reference dùng để kiểm chứng claim. Trả lời câu hỏi: Trong thực tế có đúng với prediction này không. Con người có thể tự định nghĩa "Ground truth", có thể sai sự thật.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
class_name = "person",
score = 0.912625,
bbox_xyxy = [385.33, 69.24, 498.92, 348.92], 
bbox_width = 113.58,
bbox_height = 279.68
- Diễn giải vị trí box bằng lời: Người được phát hiện nằm ở vị trí phải giữa của ảnh. Bounding box kéo dài từ khoảng x = 385 đến 499 pixel, y = 69 đến 349 pixel.
- So sánh số prediction ở hai threshold:
Threshold = 0.2 có 17 vật thể. Model chấp nhận nhiều prediction có score tương đối thấp.
Threshold = 0.35 có 11 vật thể, mất 6 vật thể. Reviewer ít phải xem những prediction yếu nhưng có thể bỏ mất object thật nhưng có score thấp.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Threshold thấp giúp giữ lại nhiều candidate detection hơn, do đó tăng cơ hội bao phủ các object thật. Nhưng đồng thời cũng tăng khổi lượng reviewer cần xem.
- Đề xuất một quy tắc box chặt:
Box 1 hình chữ nhật với cạnh bên phải có hoành độ = pixel bên phải cùng được detected là thuộc object; tương tự với cạnh phải, cạnh trên có tung độ = pixel bên trên cùng được detected là thuộc object, tương tự với cạnh dưới cùng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?
- `instance_id` dùng để làm gì và không phải loại ID nào?
- Đề xuất một quy tắc biên mask:
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Box phần nào: visible portion hay estimated full object
Khi object bị che bao nhiêu phần trăm thì annnotate
Object như nào là nhỏ (số lượng pixel, ...) và có annotate không.
Object chạm mép ảnh thì có annotate không, nếu có box phải clip theo image boundary.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |

| Phân loại ảnh | Image-level label; một class cho single-label hoặc tập class cho multi-label | Ảnh mơ hồ, nhiều chủ thể, class khó phân biệt, không có class phù hợp, label không nhất quán giữa annotator | Xem toàn ảnh, áp dụng taxonomy + guideline, chọn class phù hợp; đánh dấu ambiguous/escalate nếu không đủ căn cứ | Kiểm tra class có đúng taxonomy/guideline không; kiểm tra ảnh nhiều chủ thể, ảnh mơ hồ, disagreement và các case cần escalation |

| Phát hiện vật thể | Một record cho mỗi object instance: class + bounding box (thường xyxy, xywh hoặc format quy định) | Bỏ sót object, false object, sai class, box quá rộng/hẹp, box lệch object, object bị che/cắt mép, object quá nhỏ | Xác định từng instance, gán class, vẽ box theo quy tắc; áp dụng quy tắc occlusion/crop; escalation nếu không chắc | Kiểm tra số lượng object, class, vị trí box, độ tight của box, object bị bỏ sót/đếm trùng và các case khó |

| Instance segmentation | Một instance = một class + một segmentation mask | Mask lấn background, thiếu vùng object, mask không sát biên, các instance chồng lấn, object bị che/cắt, ranh giới không rõ | Tách từng instance, gán class, vẽ mask theo boundary thực tế và guideline; xử lý overlap/occlusion; escalation khi boundary không xác định | Kiểm tra class, số instance, boundary/mask, vùng thừa/thiếu, overlap giữa instance và tính nhất quán annotation |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng dữ liệu đúng mục đích và phạm vi được cấp quyền; không tự ý sao chép hoặc chia sẻ dữ liệu.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach, mentor

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
