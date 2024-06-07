import cupy as cp
from PIL import Image
import numpy as np
import time
import os
from tqdm import tqdm  # tqdm 라이브러리 import

def calculate_tetration(c, max_iter, threshold):
    z = cp.copy(c)
    mask = cp.ones_like(c, dtype=cp.uint8)
    for _ in range(max_iter):
        cp.power(c, z, out=z)
        diverge = cp.abs(z) > threshold
        mask[diverge] = 0  # 발산하는 픽셀을 0으로 설정
        if not cp.any(mask):  # 모든 픽셀이 발산하면 종료
            break
    return mask


def generate_frame(x_center, y_center, x_eps, image_width, image_height, max_iter, threshold, threshold_complexity):
    
    y_eps = x_eps * (image_height / image_width)

    x = cp.linspace(x_center - x_eps, x_center + x_eps, image_width)
    y = cp.linspace(y_center - y_eps, y_center + y_eps, image_height)
    X, Y = cp.meshgrid(x, y)
    C = X + 1j * Y

    # 좌표 간격이 큰 경우 데이터 타입을 축소해도 양 옆 픽셀간 구분이 가능함.
    # 좌표 간격이 작으면 작을 수록 더 정밀한 값으로 계산해야 옆 픽셀간 구분이 가능해짐.
    #  1080p 기준, 7e-4 부터 슬슬 조짐이 보이며 1e-4일 때는 일 때 많이 거슬리네요. 
    if x_eps < threshold_complexity * (image_width/1920):
        C = C.astype(cp.complex128)
    else:
        C = C.astype(cp.complex64)

    image_data = calculate_tetration(C, max_iter, threshold)
    # 수렴/발산 색을 반전시킴. 
    inverted_image_data = 1 - image_data
    return inverted_image_data

def interpolate(start, end, num_steps): # 리니어한 이동에 사용
    return [start + (end - start) * i / (num_steps - 1) for i in range(num_steps)]

def interpolate_log(start, end, num_steps): # 로그 스케일로 줌인 속도를 균일하게 유지
    log_start = np.log10(start)
    log_end = np.log10(end)
    return [10 ** (log_start + (log_end - log_start) * i / (num_steps - 1)) for i in range(num_steps)]

def main():
    # 사용자 입력
    # 시작 좌표, 좌우영역
    # x1, y1 = -0.5, 0
    # eps1 = 2.0

    x1, y1 = -4.086058278688595, -9.740283918520907e-10
    eps1 = 1.0

    # 끝 좌표, 좌우영역
    x2, y2 = -4.086058278688595, -9.740283918520907e-10
    eps2 = 3.6188788410385087e-08
    # 총 프레임 수 
    num_frames = 15*36
    max_iter = 500

    threshold_complexity = 7e-4 # 복소수 계산 정밀도 향상 기준, 1080p 기준
    threshold = 1e10 # abs(z) 수렴/발산 판정 기준
    image_width = 1920
    image_height = 1080

    # 저장 위치 설정 (예: /home/user/tetration_images/) 윈도우도 역슬래시 쓰지 말고 / 이용해 설정
    save_dir = "c:/tetration_images/"
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    print(f"저장 위치: {save_dir}")

    x_centers = interpolate(x1, x2, num_frames)
    y_centers = interpolate(y1, y2, num_frames)
    x_eps_values = interpolate_log(eps1, eps2, num_frames)

    if num_frames > 30*60: # 1800 프레임 이상 생성할 때 샘플 이미지 생성 
        # 샘플 프레임 생성
        num_sample_frames = 10
        sample_dir = os.path.join(save_dir, "sample")
        os.makedirs(sample_dir, exist_ok=True)  # sample 디렉토리 생성
        sample_x_centers = interpolate(x1, x2, num_sample_frames)
        sample_y_centers = interpolate(y1, y2, num_sample_frames)
        sample_x_eps_values = interpolate(eps1, eps2, num_sample_frames)
        for i in range(num_sample_frames):
            sample_frame_data = generate_frame(sample_x_centers[i], sample_y_centers[i], sample_x_eps_values[i], image_width, image_height, max_iter, threshold, threshold_complexity)
            sample_frame_image = Image.fromarray(cp.asnumpy(sample_frame_data * 255))
            sample_frame_image.save(os.path.join(sample_dir, f"sample_frame_{i+1:03d}.png"))

    start_time = time.time()  # 시작 시간 기록

    for i in tqdm(range(num_frames), desc="Generating frames", unit="frame"):
        frame_data = generate_frame(x_centers[i], y_centers[i], x_eps_values[i], image_width, image_height, max_iter, threshold, threshold_complexity)
        
        frame_image = Image.fromarray(cp.asnumpy(frame_data * 255))

        frame_number = i + 1
        elapsed_time = time.time() - start_time
        remaining_time = (num_frames - frame_number) * elapsed_time / frame_number
        tqdm.write(f"Frame {frame_number}/{num_frames} - Elapsed: {elapsed_time:.2f}s, Estimated Remaining: {remaining_time:.2f}s")

        save_path = os.path.join(save_dir, f"frame_{frame_number:06d}.png")
        frame_image.save(save_path)
    
    print(f"All Process is done. Image saved to: {save_path}")

if __name__ == "__main__":
    main()
