import cupy as cp
import time

def calculate_tetration_with_out(c, max_iter, threshold):
    z = cp.copy(c)
    mask = cp.ones_like(c, dtype=cp.uint8)
    start_time = time.time()
    for _ in range(max_iter):
        cp.power(c, z, out=z)
        diverge = cp.abs(z) > threshold
        mask[diverge] = 0  # 발산하는 픽셀을 0으로 설정
        if not cp.any(mask):  # 모든 픹셀이 발산하면 종료
            break
    end_time = time.time()
    return mask, end_time - start_time

def calculate_tetration_without_out(c, max_iter, threshold):
    z = cp.copy(c)
    mask = cp.ones_like(c, dtype=cp.uint8)
    start_time = time.time()
    for _ in range(max_iter):
        z = cp.power(c, z)
        diverge = cp.abs(z) > threshold
        mask[diverge] = 0  # 발산하는 픽셀을 0으로 설정
        if not cp.any(mask):  # 모든 픹셀이 발산하면 종료
            break
    end_time = time.time()
    return mask, end_time - start_time

# 테스트용 매개변수 설정
size = 1000  # 1000 x 1000 크기의 복소수 배열
max_iter = 1000
threshold = 1e6

# 복소수 배열 생성
c = cp.random.random((size, size)) + 1j * cp.random.random((size, size))

# `out` 사용 테스트
mask_out, time_out = calculate_tetration_with_out(c, max_iter, threshold)
print(f'cp.power(c, z, out=z): {time_out:.2f}초')

# `out` 미사용 테스트
mask_without_out, time_without_out = calculate_tetration_without_out(c, max_iter, threshold)
print(f'z = cp.power(c, z): {time_without_out:.2f}초')
