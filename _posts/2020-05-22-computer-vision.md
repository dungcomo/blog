---
layout: post
title:  "Thị giác máy tính - Trích xuất các đặc điểm của ảnh"
author: hoando
categories: [ computer science ]
tags: [ Vietnamese ]
image: assets/images/computer-cience.jpg
rating: 4.5
---

**Overview** <br/>
Thị giác máy tính được dùng để nghiên cứu ,thiết kế , tính toán xử lý hình ảnh. Để làm được như vậy thường sử dụng máy học .Nói cách khác Thị giác máy tính là một lĩnh vực được nghiên cứu phát triển các kỹ thuật để máy tính có thể nhìn thấy và hiểu nội dung của hình ảnh kỹ thuật số như ảnh và video<br/>

<div align="center">
    <img src="https://miro.medium.com/max/700/1*mm4Ph4YSlEVTSN2znvU1pg.jpeg" >
</div>
Một vấn đề chưa được giải quyết của thị giác máy tính là vấn đề về hạn chế tầm nhìn sinh học và sự vấn động ,phát triển phức tạp của thế giới

##  Thị giác máy tính là gì ? <br/>
Ở mức độ nào đó , thị giác máy tính dùng những hình ảnh mà nó quan sát được để suy luận điều gì đó về thế giới <br/>

<div align="center">
    <img src="https://miro.medium.com/max/365/1*AxHTaOiHA2rb5Q2XdHzYGw.png" >
    <p>(Mối quan hệ giữa AI,ML,CV)</p>
</div>

_Lưu ý : Thị giác máy tính khác với xử lý ảnh_

Xử lý hình ảnh (Image processing) là quá trình tạo ra một hình ảnh mới từ hình ảnh đã có , thường tối ưu hóa hoặc nâng cao nội dung của bức ảnh theo một cách nào đó . Nó là một loại xử lý tín hiệu số và không liên quan đến việc hiểu nội dung của hình ảnh

Một hệ thống thị giác máy tính có thể yêu cầu xử lý hình ảnh được áp dụng với hình ảnh nguyên bản ,ví dụ : preprocessing ( tiền xử lý )
## Một số phương pháp xử lý ảnh phổ biến
Trước khi tiếp cận với các phương pháp, chúng ta cần có một số khái niệm sau:
**Ảnh kĩ thuật số (Digital image)**
Ảnh kĩ thuật số (digital image) là một dạng biểu diễn của ảnh ở dạng ma trận số 2 chiều. Tùy vào độ phân giải của ảnh có cố định hay không, ảnh kĩ thuật số được chia ra làm 2 loại là ảnh vector (độ phân giải không cố định) và ảnh raster (hay còn gọi là bitmapped, độ phân giải cố định). Thuật ngữ ảnh kĩ thuật số thường được dùng để nói đến ảnh raster.

**Độ phân giải của ảnh (Resolution)**
Độ phân giải ảnh là mức độ chi tiết mà ảnh có thể thể hiện. Thuật ngữ này được dùng cho ảnh raster. Độ phân giải càng cao, ảnh càng nhiều chi tiết. Ví dụ: độ phân giải 640 x 480.

**Điểm ảnh (Pixel)**
Trong ảnh kĩ thuật số, một điểm ảnh (pixel) là phần tử nhỏ nhất của ảnh raster (raster image). Mỗi một điểm ảnh là một mẫu (sample) của ảnh. Càng nhiều điểm ảnh, ảnh kĩ thuật số càng biểu diễn chính xác hơn về nội dung của ảnh gốc. Đặc trưng của một điểm ảnh gồm 2 thành phần: tọa độ (x,y) và cường độ sáng (intensity).

**Mức xám của ảnh (Grayscale)**
Mức xám của ảnh (greyscale) là một trong những giá trị số của điểm ảnh biểu diễn mức độ ánh sáng (light intensity) tại điểm ảnh đấy. Thông thường, trong xử lý ảnh hiện tại, mức xám hay sử dụng nhất là mức 256 (có giá trị mức xám từ 0 -> 255).

**Ảnh màu**
Để biểu diễn ảnh màu, theo cách biểu diễn RGB, ba ma trận mức xám 256, ứng với ba màu đỏ (R), lục (G), lam(B) được sử dụng. Màu sắc của một điểm ảnh được quyết định bởi giá trị cường độ (intensity) tại ba ma trận màu cùng tọa độ.

# Triết xuất các đặc tính của ảnh thông qua cường độ điểm ảnh

Ta có thể thấy, một hình ảnh cũng xem như một ma trận, trong đó mỗi điểm ảnh đại diện cho một màu sắc và một  _thứ nào đó(được gọi là vector đặc trưng)_ có thể bao quát được toàn bộ đặc tính của ảnh được xây dựng bằng cách định hình lại ma trận thành một vector ( hay có thể nói , chúng ta sẽ ghép các hàng của ma trận lại với nhau — làm phẳng ma trận)

Nhận dạng ký tự quang học ORC( Optical Character Recognition) là một vấn đề về machine learning kinh điển và chữ số viết tay là một ví dụ kinh điển cho vấn đề này

Bộ dữ liệu viết tay của scikit-learning chứa hình ảnh thang độ sám (Grayscale) của hơn 1700 chữ viết tay với các số từ 0 đến 9 .Mỗi bức ảnh có 8 pixels ở mỗi bên. Mỗi pixels được biểu thị giá trị cường độ từ 0 đến 16 , trắng là cường độ cao nhất biểu thị mức 0, đen là cường độ thấp nhất biểu thị mức 16

![enter image description here](https://miro.medium.com/max/323/1*t9CwRfSyRjp4Q7Y0MBNoCw.png)<br/>
(Hình ảnh biểu diễn số 0)

Bây giờ chúng ta sẽ thực hiện một số hàm trên scikit-learn để hiểu rõ vấn đề
![hình ảnh đã được chuyển sang ma trận 8x8](https://miro.medium.com/max/424/1*je7IjOo1JdqPrZjq1TD0Yw.png)<br/>
(Hình ảnh đã được chuyển sang ma trận 8x8)<br/>
Phép biểu diễn bằng ma trân trên chỉ có có thể thực hiện một số công việc cơ bản, như nhận diện ký tự . Tuy nhiên với một hình ảnh có pixels lớn thì đồng nghĩa với việc sẽ có một ma trận có kích thước lớn dẫn đến một vector đặc trưng cũng sẽ có kích thước lớn

Ví dụ : một hình ảnh thang độ xám có kích thước 100x100 sẽ cho chúng ra vector đặc trưng 10000 chiều

và một hình ảnh thang độ xám có kích thước 1920x1080 sẽ cho chúng ra vector đặc trưng là 2,073,600 chiều

kích thước lớn không phải là nhược điểm duy nhất của kỹ thuật này , việc training tại các cường độ sáng ứng các điểm pixel ở những vị trí cụ thể dẫn đến việc model sau khi training sẽ bị nhạy cảm với một số yếu tố như thay đổi tỷ lệ, xoay và dịch ảnh

Một model sau khi đã training sẽ không nhận diện được số 0 trong ví dụ trước khi dịch vài pixels theo bất kỳ hướng nào ,phóng to, xoay, một vài độ

Hơn nữa việc học hỏi từ cường độ điểm ảnh là một vấn đề vì model dễ nhảy cảm với những thay đổi về chiếu sáng

Vì những lý do này , kỹ thuật này không hiệu quả đối với các tác vụ liê n quan bao gồm ảnh chụp hoặc ảnh tự nhiên

# Triết xuất các đặc tính của ảnh thông qua các điểm đáng chú ý (points of interest)

Với kỹ thuật “_Triết xuất các đặc tính của ảnh thông qua cường độ điểm ảnh”_ vector đặc trưng bao gồm các giá trị đặc tính hình ảnh và các giá trị nhiễu sau khi quan sát dữ liệu sau khi thực hiện xong preprocessing ta thấy giá trị biểu diễn pixels trắng thực sự không hữu ích cũng như con người có thể nhận ra được đối tượng nhưng không cần quan sát toàn bộ thuộc tính của đối tượng . Vì vậy _kỹ thuật này_  sẽ chỉ ra đặc tính đặc trưng của hình ảnh.

Cạnh(**Edges**) và góc (**corners**) là 2 loại points of interest.(PI) phổ biến

**Edges**  là một ranh giới mà tại đó cường độ điểm ảnh thay đổi nhanh chóng và một  **Corners** được tạo bởi từ giao điểm 2 cạnh (**Edges**)