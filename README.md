# Thế Giới Hải Tặc (Super Pirate World)

![Logo Game](images/logo.png)

"Thế Giới Hải Tặc" là một game platformer 2D được phát triển bằng Python với thư viện Pygame. Người chơi nhập vai một hải tặc vượt qua các màn chơi đầy thử thách, thu thập xu, chiến đấu với kẻ thù, và chinh phục các level để khám phá thế giới rộng lớn.

## Mục lục
- [Trạng thái bắt đầu](#trạng-thái-bắt-đầu)
- [Trạng thái kết thúc](#trạng-thái-kết-thúc)
- [Chi phí](#chi-phí)
- [Hành động](#hành-động)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Cấu trúc file](#cấu-trúc-file)
- [Đóng góp](#đóng-góp)

---

## Trạng thái bắt đầu

Người chơi bắt đầu hành trình trong game với các điều kiện sau:  
- **Vị trí ban đầu**: Ở màn hình **Overworld** tại **node 0** hoặc trong **level 0** tại tọa độ do bản đồ Tiled xác định.  
- **Thông số ban đầu**:  
  - **Máu**: 5  
  - **Xu**: 0  
  - **Level mở khóa**: Chỉ level 0 khả dụng  

**Cấu trúc lưu trữ**:  
Lớp `Data` quản lý thông tin nhân vật: máu, xu, cấp hiện tại, vị trí nhân vật.

---

## Trạng thái kết thúc

Game kết thúc khi một trong các điều kiện sau xảy ra:  
- **Nhân vật mất hết máu**: Máu về 0, game thoát.  
- **Nhân vật di chuyển ra khỏi map**: Rơi khỏi đáy bản đồ, mất 1 máu và quay lại Overworld.  
- **Nhân vật đi đến đích**: Chạm cờ hải tặc, hoàn thành level và trở lại Overworld.

**Cấu trúc lưu trữ**:  
Lớp `Data` theo dõi máu, xu, cấp hiện tại, và vị trí nhân vật.

---

## Chi phí

Chi phí trong game được hiểu là tài nguyên hoặc nỗ lực mà người chơi phải bỏ ra để hoàn thành mục tiêu:  
- **Lượng máu mất đi**:  
  - Mỗi lần bị kẻ thù hoặc chướng ngại vật tấn công, người chơi mất 1 máu.  
  - Chi phí tối thiểu: 0 (nếu không bị tổn thương).  
  - Chi phí tối đa: 5 (mất hết máu, dẫn đến thua).  
- **Xu như tài nguyên bổ trợ**:  
  - Thu thập xu từ vật phẩm để đổi máu (100 xu = 1 máu).  
  - Không bắt buộc để thắng, nhưng ảnh hưởng đến khả năng sống sót.

---

## Hành động

Người chơi có thể thực hiện các hành động sau để điều khiển nhân vật:  
- **Di chuyển**:  
  - **Trái**: Nhấn phím LEFT (`pygame.K_LEFT`).  
  - **Phải**: Nhấn phím RIGHT (`pygame.K_RIGHT`).  
  - **Nhảy**: Nhấn phím SPACE (`pygame.K_SPACE`), có thể nhảy tường nếu chạm tường trái/phải.  
- **Tấn công**:  
  - **Chém**: Nhấn phím X (`pygame.K_x`), gây ảnh hưởng đến kẻ thù trong phạm vi gần.  

![Gameplay Screenshot](images/screenshot.png)

---

## Hướng dẫn cài đặt

1. **Yêu cầu**:  
   - Python 3.8+  
   - Thư viện: `pygame`, `pytmx`  
   - Công cụ tạo bản đồ: Tiled Map Editor (để chỉnh sửa file `.tmx`).  

2. **Cài đặt**:  
   ```bash
   pip install pygame pytmx
