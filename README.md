**THẾ GIỚI HẢI TẶC**
**Trạng thái bắt đầu**
-	Người chơi bắt đầu ở Overworld tại node 0 hoặc trong level 0 tại vị trí do bản đồ Tiled xác định, với 5 máu, 0 xu, và chỉ level 0 được mở khóa.
 
***Cấu trúc lưu trạng thái bắt đầu:***
	Lớp data lưu trữ các thông tin của nhân vật: máu, xu, cấp hiện tại, vị trí nhân vật.
**Trạng thái kết thúc:**
-	Nhân vật mất hết máu
-	Nhân vât di chuyển ra khỏi map
-	Nhân vật đi đến đích (cờ hải tặc) trở lại overword
 
***Cấu trúc lưu trữ***
	Lớp data lưu trữ các thông tin của nhân vật: máu, xu, cấp hiện tại, vị trí nhân vật.
**Chi phí:**
	Lượng máu mất đi khi qua một màng chơi
**Hành động:**
Di chuyển: trái, phải, nhảy
Tấn công: chém.
