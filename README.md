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
- **Vị trí ban đầu**:  
  - Ở màn hình **Overworld** tại **node 0** (điểm xuất phát trên bản đồ chọn level).  
  - Nếu vào level ngay, nhân vật xuất hiện tại tọa độ `(obj.x, obj.y)` do bản đồ Tiled xác định (layer "Objects", `obj.name == 'player'`).  
- **Thông số ban đầu**:  
  - **Máu**: 5 (người chơi có 5 mạng sống).  
  - **Xu**: 0 (không có xu khi bắt đầu).  
  - **Level mở khóa**: Chỉ level 0 khả dụng (`self.unlocked_level = 0`).  
  - **Trạng thái nhân vật**: Đứng yên (`self.state = 'idle'`), hướng mặt sang phải (`self.facing_right = True`).  

**Cấu trúc lưu trữ trạng thái bắt đầu**:  
- Lớp `Data` (`data.py`) quản lý thông tin toàn cục:  
  - `self._health`: Lượng máu (5).  
  - `self._coins`: Số xu (0).  
  - `self.current_level`: Level hiện tại (0).  
  - `self.unlocked_level`: Level đã mở khóa (0).  
- Lớp `Player` (`player.py`) lưu thông tin nhân vật trong level:  
  - `self.rect`: Vị trí hình chữ nhật hiển thị.  
  - `self.hitbox_rect`: Vị trí va chạm.  
  - `self.state`: Trạng thái hoạt ảnh (`idle`).  
  - `self.direction`: Vector di chuyển (`vector(0, 0)`).  
- Lớp `Overworld` (`overworld.py`) lưu vị trí icon:  
  - `self.icon.rect`: Tọa độ icon tại node 0.

---

## Trạng thái kết thúc

Game kết thúc khi một trong các điều kiện sau xảy ra:  
- **Nhân vật mất hết máu**:  
  - Khi `self.data.health <= 0`, game thoát hoàn toàn.  
- **Nhân vật di chuyển ra khỏi map**:  
  - Nếu nhân vật rơi xuống dưới đáy bản đồ (`self.player.hitbox_rect.bottom > self.level_bottom`), mất 1 máu và quay lại Overworld.  
- **Nhân vật đi đến đích**:  
  - Khi nhân vật chạm cờ hải tặc (`self.level_finish_rect`), level hoàn thành, quay lại Overworld và mở khóa level tiếp theo.

**Cấu trúc lưu trữ trạng thái kết thúc**:  
- Lớp `Data` (`data.py`) cập nhật thông tin:  
  - `self._health`: Máu còn lại (0 nếu thua, >0 nếu thắng).  
  - `self._coins`: Số xu tích lũy.  
  - `self.current_level`: Level vừa hoàn thành.  
  - `self.unlocked_level`: Level tối đa đã mở khóa.  
- Lớp `Player` (`player.py`) lưu trạng thái cuối:  
  - `self.rect` và `self.hitbox_rect`: Vị trí khi chạm đích hoặc rơi khỏi map.  
  - `self.state`: Trạng thái cuối cùng (ví dụ: `fall` nếu rơi, `idle` nếu thắng).  
- Lớp `Level` (`level.py`) xác định kết quả:  
  - `self.level_finish_rect`: Vùng đích để kiểm tra thắng.  
  - `self.level_bottom`: Giới hạn dưới để kiểm tra thua.

---

## Chi phí

Chi phí trong game là tài nguyên hoặc nỗ lực mà người chơi bỏ ra để hoàn thành mục tiêu:  
- **Lượng máu mất đi**:  
  - Mỗi lần bị kẻ thù hoặc chướng ngại vật tấn công (`hit_collision` trong `level.py`), mất 1 máu.  
  - Chi phí tối thiểu: 0 (nếu không bị tổn thương).  
  - Chi phí tối đa: 5 (mất hết máu, dẫn đến thua).  
- **Xu như tài nguyên bổ trợ**:  
  - Thu thập xu từ vật phẩm (`Item` trong `sprites.py`) để đổi máu (100 xu = 1 máu, xử lý trong `data.py`).  
  - Không bắt buộc để thắng, nhưng ảnh hưởng đến khả năng sống sót.

---

## Hành động

Người chơi có thể thực hiện các hành động sau để điều khiển nhân vật:  
- **Di chuyển**:  
  - **Trái**: Nhấn phím LEFT (`pygame.K_LEFT`).  
  - **Phải**: Nhấn phím RIGHT (`pygame.K_RIGHT`).  
  - **Nhảy**: Nhấn phím SPACE (`pygame.K_SPACE`), hỗ trợ nhảy tường nếu chạm tường trái/phải.  
- **Tấn công**:  
  - **Chém**: Nhấn phím X (`pygame.K_x`), gây ảnh hưởng đến kẻ thù trong phạm vi gần (`attack_collision` trong `level.py`).  

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
