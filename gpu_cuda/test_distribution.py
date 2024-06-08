import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# 해상도 설정
resolution_x, resolution_y = 1920, 1080  # 샘플링 크기를 늘림
resolution_y_ratio = resolution_y / resolution_x

# 범위 설정
eps = 9
real_min, real_max = 4 - eps, 4 + eps
imag_min, imag_max = 0 - eps * resolution_y_ratio, 0 + eps * resolution_y_ratio

# x, y 좌표 생성
real_values = cp.linspace(real_min, real_max, resolution_x)
imag_values = cp.linspace(imag_min, imag_max, resolution_y)

# 그리드 생성
real_grid, imag_grid = cp.meshgrid(real_values, imag_values)
complex_grid = real_grid + 1j * imag_grid

# 테트레이션 계산 함수 정의
def tetration(c, n):
    result = c
    for _ in range(n - 1):
        result = c ** result
        if cp.any(cp.abs(result) > 1e10):  # 오버플로우 방지
            result = cp.where(cp.abs(result) > 1e10, complex('inf'), result)
    return result

# 벡터화된 테트레이션 계산
n = 20  # 테트레이션의 횟수
magnitudes = cp.abs(tetration(complex_grid, n))
print(f'calculate tetration finished')

# CuPy 배열을 NumPy 배열로 변환 및 무한대를 제외한 값만 추출
finite_magnitudes = magnitudes[cp.isfinite(magnitudes)].get()

# 통계적 지표 계산
mean_magnitude = np.mean(finite_magnitudes)
variance_magnitude = np.var(finite_magnitudes)
std_dev_magnitude = np.std(finite_magnitudes)
min_magnitude = np.min(finite_magnitudes)
max_magnitude = np.max(finite_magnitudes)
median_magnitude = np.median(finite_magnitudes)
percentile_25 = np.percentile(finite_magnitudes, 25)
percentile_50 = np.percentile(finite_magnitudes, 50)
percentile_75 = np.percentile(finite_magnitudes, 75)
print(f'calculate statistics finished')

# 추가 정보 계산
total_pixels = resolution_x * resolution_y
num_infinite = total_pixels - len(finite_magnitudes)
num_finite = len(finite_magnitudes)

# 결과 출력
stats = {
    "Mean": mean_magnitude,
    "Variance": variance_magnitude,
    "Standard Deviation": std_dev_magnitude,
    "Min": min_magnitude,
    "Max": max_magnitude,
    "Median": median_magnitude,
    "25th Percentile": percentile_25,
    "50th Percentile": percentile_50,
    "75th Percentile": percentile_75,
    "Total Pixels": total_pixels,
    "Excluded (inf)": num_infinite,
    "Analyzed": num_finite,
}

print(stats)


# 결과값의 분포를 플롯

# plt.hist(finite_magnitudes, bins=50, edgecolor='k', alpha=0.7)
plt.hist(finite_magnitudes, bins=50, edgecolor='k', alpha=0.7, log=True) # 로그 스케일로 
plt.title('Distribution of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True)

# 통계값을 플롯에 표시
textstr = '\n'.join((
    f"Mean: {mean_magnitude:.2f}",
    f"Variance: {variance_magnitude:.2f}",
    f"Std Dev: {std_dev_magnitude:.2f}",
    f"Min: {min_magnitude:.2e}",
    f"Max: {max_magnitude:.2e}",
    f"Median: {median_magnitude:.2f}",
    f"25th Percentile: {percentile_25:.2f}",
    f"50th Percentile: {percentile_50:.2f}",
    f"75th Percentile: {percentile_75:.2f}",
    f"Total Pixels: {total_pixels}",
    f"Excluded (inf): {num_infinite}",
    f"Analyzed: {num_finite}"
))

# 텍스트 박스의 속성
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# 텍스트 박스를 플롯에 추가
plt.gcf().text(0.35, 0.85, textstr, fontsize=12, verticalalignment='top', bbox=props)

plt.show()
