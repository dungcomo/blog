---
layout: post
title:  "Sử dụng thuật toán KNN trong bài toán Classification"
author: hoando
categories: [ Algorithms ]
tags: [ Machine learning ]
image: assets/images/machine-learning-840x430.png
rating: 4.5
---
Một thuật toán kinh điển trong bài toán Classification

## Overview

- Là một thuật toán phân loại ảnh dễ triển khai
- Không học được bất cứ điều gì từ tập dữ liệu
- Chỉ dựa vào khoảng cách giữa các vector đặc trưng để phân loại tập dữ liệu<br/>
Một cách dễ hiểu , thuật toán KNN phân loại các dữ liệu chưa biết bằng cách tìm bộ dữ liệu đã biết gần nó nhất sau đó gán nhãn theo nhãn của bộ dữ liệu đã biết <br/>

<div align="center">
    <img src="https://miro.medium.com/max/700/1*cEgY1t09bzDf3EsVfxtOPA.png"/>
<p>
(K-NN classfication)
</p>
</div>
mẫu thử của chúng ta làm chấm tròn xanh lá phải được gán nhãn là hình vuông xanh hoặc hình tam giác đỏ. <br/>
- Xét K = 3 thì nó được gán nhãn là tam giác màu đỏ ( xét hình tròn nét liền)
- Xét K = 5 thì nó được gán nhãn là hình vuông màu xanh ( xét hình tròn nét đứt )

###  Ví dụ sau đây mô tả quá trình thực hiện thuật toán KNN
**trước hết ta tạo ra bộ dữ liệu các đối tượng và gán nhãn cho chúng**
```python
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
label = np.random.randint(0,2,(25,1)).astype(np.float32)

squares = trainData[label.ravel() == 1]
triangles = trainData[label.ravel() == 0]

```
> trainData là một ma trận 25x2 (25 hàng ,2 cột ) được tạo ra bởi hàm np.random.randint()trainData là một ma trận 25x2 (25 hàng ,2 cột ) được tạo ra bởi hàm np.random.randint()
- Với 0,100 : là các giá trị được lấy nằm trong [0,100)
- (25,2) : là kiểu ma trận 25x2
- astype() : ép kiểu dữ liệu trong ma trận về dạng số thực
- label : là nhãn được đánh số 0 và 1
- Những giá trị gán nhãn 0 được coi là hình tam giác
- Những giá trị gán nhãn 1 được coi là hình vuông
- hàm ravel() được dùng để làm phẳng ma trận , hay duỗi ma trận về ma trận 1 chiều

**phân tích** : trainData[label.ravel() == 1]
- label.ravel() == 1 sẽ trả về một list có giá trị true hoặc false
- tại vị trí mà trainData[True] thì sẽ được đẩy vào squares
như vậy ta đã gán nhãn được những bộ giá trị trong trainData

```python
targetPredict = np.random.randint(0,100,(1,2)).astype(np.float32)

plt.scatter(triangles[:0],triangles[:,1],100,'r','^')
plt.scatter(squares[:,0],squares[:,1],100,'b','s')
plt.scatter(targetPredict[:,0],targetPredict[:,1],100,'g','o')

```
sau đó tạo ra 1 giá trị targetPredict cần dự đoán
<div align="center">
    <img src="https://miro.medium.com/max/700/1*xLHJH7w-vVeUv21r4fENEg.png"/>
    <p>hình vuông : 1, hình tam giác : 0</p>
</div>

quan sát ta thấy chấm tròn là điểm ta cần gán nhãn cho nó là hình vuông hay trình tam giác

```python
knn =cv2.ml.Knearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,label)
temp,result, nearest, distance = knn.findNearest(targetPredict,3)

print("Target to predict",targetPredict)
print("Result target to predict : ",format(result))
print("nearest with:",format(nearest))
print("distance :",format(distance))

```
Xét với trường hợp k = 3 ta có kết quả sau:
- target to predict : [[38. 3.]]
- result target to predict : [[1.]]
- nearest with : [[1. 0. 1.]]
- distance is : [[ 20. 362. 392.]]
<br/>
kết quả đúng với quan sát thực tế trên đồ thị

> Lưu ý : nên xét các trường hợp k lẻ thì thuật toán sẽ dễ dàng giải quyết hơn

