# Thế Giới Hải Tặc (Super Pirate World)

## Trạng thái bắt đầu

Người chơi bắt đầu hành trình trong game với các điều kiện sau:  
- **Vị trí ban đầu**: Ở màn hình **Overworld** tại **node 0** hoặc trong **level 0** tại tọa độ do bản đồ Tiled xác định.  
- **Thông số ban đầu**:  
  - **Máu**: 5  
  - **Xu**: 0  
  - **Level mở khóa**: Chỉ level 0 khả dụng  

**Cấu trúc lưu trữ**:  
Lớp `Data` quản lý thông tin nhân vật: máu, xu, cấp hiện tại, vị trí nhân vật.

## Trạng thái kết thúc

Game kết thúc khi một trong các điều kiện sau xảy ra:  
- **Nhân vật mất hết máu**: Máu về 0, game thoát.  
- **Nhân vật di chuyển ra khỏi map**: Rơi khỏi đáy bản đồ, mất 1 máu và quay lại Overworld.  
- **Nhân vật đi đến đích**: Chạm cờ hải tặc, hoàn thành level và trở lại Overworld.

**Cấu trúc lưu trữ**:  
Lớp `Data` theo dõi máu, xu, cấp hiện tại, và vị trí nhân vật.
