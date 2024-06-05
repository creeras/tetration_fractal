import cupy as cp
import numpy as np  # numpy는 해상도 설정 등 일부 CPU 연산에 사용될 수 있습니다.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import os

class TetrationFractalExplorer:
    def __init__(self):
        # tkinter 윈도우 초기 설정
        self.root = tk.Tk()
        self.root.title("Tetration Fractal Cuda Explorer")

        # 객체 설정, 초기회에서 빼고 함수파트에만 넣으려고 했는데, 에러나서 다시 추가.
        self.fig, self.ax = plt.subplots()   # figsize를 설정하지 않음
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 메뉴 바 설정
        self.menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_exit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=self.menu_bar)

        # 상태 바 설정
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")

        # 초기 프랙탈 경계 설정
        self.center_x, self.center_y = -0.5, 0
        self.eps = 1

        # 최대 반복 횟수 설정
        self.max_iter_var = tk.StringVar()
        self.max_iter_var.set("100")  # 초기값 설정

        # 플롯 화면 비율 설정
        self.ratio_options = ["4K", "1080p", "720p", "1080(1:1)"]
        self.ratio_var = tk.StringVar(value=self.ratio_options[1])  # 기본값을 1080p로 설정.

        # rect 속성 초기화
        self.rect = None

        # 설정 프레임
        self.settings_frame = ttk.Frame(self.root)
        self.settings_frame.pack(side=tk.TOP, fill=tk.X)

        # Set up the controls in the settings frame
        self.setup_controls()

        # Generate the initial fractal
        self.generate_fractal()

        # Set up rectangle selector tool
        self.toggle_selector = RectangleSelector(self.ax, self.on_select, interactive=True, useblit=True,
                                                 button=[1], minspanx=5, minspany=5, spancoords='pixels')

        # Start the main loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)  # 윈도우 닫기 버튼 클릭 시 호출되는 함수 설정
        self.root.mainloop()

    def setup_controls(self):
        """Sets up the controls in the settings frame."""
        ttk.Label(self.settings_frame, text="Max Iterations:").pack(side=tk.LEFT, padx=10)

        # 최대 반복 횟수 옵션 설정
        self.max_iter_options = [20, 50, 100, 200, 500]
        self.max_iter_var = tk.StringVar(value=str(self.max_iter_options[2]))
        self.max_iter_dropdown = ttk.OptionMenu(self.settings_frame, self.max_iter_var, str(self.max_iter_options[2]), *map(str, self.max_iter_options), command=self.update_fractal)
        self.max_iter_dropdown.pack(side=tk.LEFT, padx=10)

        # 플롯 화면 비율 옵션 설정
        ttk.Label(self.settings_frame, text="Resolution:").pack(side=tk.LEFT, padx=10)
        self.ratio_dropdown = ttk.OptionMenu(self.settings_frame, self.ratio_var, self.ratio_options[2], *self.ratio_options, command=self.update_ratio)
        self.ratio_dropdown.pack(side=tk.LEFT, padx=10)

        # 좌표 및 eps 값 입력 설정
        ttk.Label(self.settings_frame, text="x=").pack(side=tk.LEFT, padx=5)
        self.x_entry = ttk.Entry(self.settings_frame, width=17)
        self.x_entry.pack(side=tk.LEFT, padx=5)
        self.x_entry.insert(0, f"{self.center_x:.14f}")

        ttk.Label(self.settings_frame, text="y=").pack(side=tk.LEFT, padx=5)
        self.y_entry = ttk.Entry(self.settings_frame, width=17)
        self.y_entry.pack(side=tk.LEFT, padx=5)
        self.y_entry.insert(0, f"{self.center_y:.14f}")

        ttk.Label(self.settings_frame, text="eps=").pack(side=tk.LEFT, padx=5)
        self.eps_entry = ttk.Entry(self.settings_frame, width=17)
        self.eps_entry.pack(side=tk.LEFT, padx=5)
        self.eps_entry.insert(0, f"{self.eps:.14f}")

        ttk.Button(self.settings_frame, text="Apply", command=self.apply_coordinates).pack(side=tk.LEFT, padx=10)

        # 줌 인/아웃 버튼 설정
        ttk.Label(self.settings_frame, text="Zoom:").pack(side=tk.LEFT, padx=10)
        self.zoom_options = ["x10000", "x1000", "x100", "x10", "x2", "1/2", "1/10", "1/100", "1/1000", "1/10000"]
        self.zoom_var = tk.StringVar(value=self.zoom_options[0])
        self.zoom_dropdown = ttk.OptionMenu(self.settings_frame, self.zoom_var, self.zoom_options[0], *self.zoom_options, command=self.zoom)
        self.zoom_dropdown.pack(side=tk.LEFT, padx=10)

        # 리셋 버튼 설정
        ttk.Button(self.settings_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=10)

        # 이미지 저장 버튼 설정
        ttk.Button(self.settings_frame, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=10)

    def tetration(self, z, max_iter):
        """Computes the tetration fractal using GPU acceleration."""
        result = cp.zeros(z.shape, dtype=cp.complex128) + 1 # 초기값을 1로 세팅할 때 +1
        for _ in range(max_iter):
            try:
                result = cp.exp(result * cp.log(z))
            except OverflowError:
                result = cp.inf
        return result

    def generate_fractal(self):
        """Generates and displays the fractal. 플롯 이미지 생성"""

        self.update_status("Generating fractal...")

        self.regenerate_canvas()
        max_iter = int(self.max_iter_var.get())
        Z, x_range, y_range = self.calculate_plot_range(max_iter)

        self.fractal = self.tetration(Z, max_iter)  # self.fractal에 저장

        self.plot_fractal(max_iter, x_range, y_range) # plot_fractal_alter 를 사용해 볼 수도 있으나... 별로인듯?

    def regenerate_canvas(self):
        # self.fig, self.ax, self.canvas 객체 삭제
        # 원래는 불필요한 과정이 맞으나, 
        # 이상하게 컬러바가 삭제되지 않고 추가생성되며 플롯 영역이 줄어드는 문제가 있어서 
        # 캔버스를 삭제하고 새로 그리기로 함. 

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        # if self.ax is not None: 
        #    self.ax = None

        # matplotlib, 캔버스 설정
        # self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 초기 figsize 설정
        self.fig, self.ax = plt.subplots()   # figsize를 설정하지 않음
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Set up rectangle selector tool
        self.toggle_selector = RectangleSelector(self.ax, self.on_select, interactive=True, useblit=True,
                                                 button=[1], minspanx=5, minspany=5, spancoords='pixels')

    def calculate_plot_range(self, max_iter):
        # 이미지 플롯할 해상도와 비율 얻기
        resolution_options = {
        "4K": (3840, 2160),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "1080(1:1)": (1080, 1080),
        }
        selected_resolution = self.ratio_var.get()
        if selected_resolution in resolution_options:
            self.plot_width, self.plot_height = resolution_options[selected_resolution]
        else:
            self.plot_width, self.plot_height = 1280, 720  # 기본값 설정

        self.aspect_ratio = self.plot_width / self.plot_height

        # x와 y의 범위 설정
        x_range = 2 * self.eps
        y_range = x_range / self.aspect_ratio
        
        # 중심점&eps로 플롯할 사각형 범위(range)를 얻음
        x = cp.linspace(self.center_x - x_range / 2, self.center_x + x_range / 2, self.plot_width)
        y = cp.linspace(self.center_y - y_range / 2, self.center_y + y_range / 2, self.plot_height)
        X, Y = cp.meshgrid(x, y)
        Z = X + 1j * Y

        return Z, x_range, y_range # 함수 분리로 반환 필요. 

    def plot_fractal(self, max_iter, x_range, y_range):
        if self.eps > 0:
            self.ax.clear()
            extent = self.center_x - x_range / 2, self.center_x + x_range / 2, self.center_y - y_range / 2, self.center_y + y_range / 2
            fractal_gpu = abs(cp.asnumpy(cp.angle(self.fractal)))
            self.ax.imshow(fractal_gpu, extent=extent, cmap='hsv', origin='lower')
            self.ax.set_xlabel("Re")
            self.ax.set_ylabel("Im")
            self.ax.set_title(f'Tetration Fractal: Iterations={max_iter}\n'
                            f'x={self.center_x:.14f}, y={self.center_y:.14f}, eps={self.eps:.14f}')
            self.ax.set_autoscale_on(False)

            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.fig.colorbar(self.ax.get_images()[0], ax=self.ax)
            self.canvas.draw()
        
            self.update_status("Fractal generated. Ready...")
        else:
            self.update_status("Warning: eps must be positive")

    def plot_fractal_alter(self, max_iter, x_range, y_range):
        magnitude = np.abs(self.fractal)
        color_magnitude = np.log(magnitude + 1e-9)

        self.ax.clear()
        img = self.ax.imshow(color_magnitude, cmap='hsv', extent=[self.center_x - x_range / 2, self.center_x + x_range / 2,
                                                                  self.center_y - y_range / 2, self.center_y + y_range / 2])

        self.fig.colorbar(img, ax=self.ax, fraction=0.046, pad=0.04)
        self.ax.set_title(f"Tetration Fractal\nMax Iterations: {max_iter}, Center: ({self.center_x:.5f}, {self.center_y:.5f}), Eps: {self.eps:.5f}")

        self.canvas.draw()

        self.update_status("Fractal generated.")




    def update_fractal(self, *args):
        """Updates the fractal based on maximum iterations.
        max_iter 리스트에서 선택 후에만 호출됨 """
        try:
            self.max_iter_var.set(args[0])  # 리스트에서 설정된 값 1개만 있음. 그래서 [0]
            self.generate_fractal()
            self.update_status(f'Updated max iterations to {self.max_iter_var.get()}')
        except ValueError:
            self.update_status("Invalid input for max iterations. Please enter a valid number.")

    def apply_coordinates(self):
        """Applies the user-input coordinates and eps."""
        try:
            self.center_x = float(self.x_entry.get())
            self.center_y = float(self.y_entry.get())
            self.eps = float(self.eps_entry.get())
            self.generate_fractal()
            self.update_status(f'Applied x={self.center_x:.14f}, y={self.center_y:.14f}, eps={self.eps:.14f}')
        except ValueError:
            self.update_status("Invalid input. Please enter valid numbers.")

    def zoom(self, *args):
        """Zooms in or out based on the selected zoom factor."""
        zoom_factor_str = self.zoom_var.get()
        zoom_factor = float(zoom_factor_str[1:]) if "x" in zoom_factor_str else 1 / float(zoom_factor_str[2:])
        self.eps /= zoom_factor
        self.eps_entry.delete(0, tk.END)
        self.eps_entry.insert(0, f"{self.eps:.14f}")
        self.generate_fractal()
        self.update_status(f'Zoomed {"in" if zoom_factor > 1 else "out"} by {zoom_factor}')

    def reset(self):
        """Resets the fractal to the initial state."""
        self.center_x, self.center_y = -0.5, 0
        self.eps = 1
        self.max_iter_var.set("100")
        self.generate_fractal()
        self.update_status("Reset to initial state")

    def save_image(self):
        """Saves the fractal image at the selected resolution.
        이미 그려진 화면 plot 이용하는게 경제적이지만, 
        화면 해상도가 아닌 선택한 resolution 에 맞는 이미지로 저장하기 위해 새로 플롯을 그림. 
        """
        folder_name = datetime.now().strftime('%Y-%m-%d')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_folder_path = os.path.join(script_dir, folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        filename = (f'x={self.center_x:.14f}_y={self.center_y:.14f}_'
                    f'eps={self.eps:.14f}_iteration={int(self.max_iter_var.get())}_'
                    f'{self.ratio_var.get().replace(":", "x")}.png')
        file_path = os.path.join(full_folder_path, filename)

        if os.path.exists(file_path):
            self.update_status(f"File already exists: {file_path}")
        else:
            fig, ax = plt.subplots(figsize=(self.plot_width / 100, self.plot_height / 100))
            x_range = 2 * self.eps
            y_range = 2 * self.eps / self.aspect_ratio
            extent = self.center_x - x_range / 2, self.center_x + x_range / 2, self.center_y - y_range / 2, self.center_y + y_range / 2
            fractal_gpu = cp.asnumpy(cp.angle(self.fractal))
            ax.imshow(fractal_gpu, extent=extent, cmap='hsv', origin='lower')

            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            ax.set_title(f'Tetration Fractal: Iterations={int(self.max_iter_var.get())}\n'
                        f'x={self.center_x:.14f}, y={self.center_y:.14f}, eps={self.eps:.14f}')
            ax.set_autoscale_on(False)

            fig.savefig(file_path, dpi=100)
            plt.close(fig)  # 메모리 해제를 위해 플롯을 닫음

            self.update_status(f"Image saved as {os.path.abspath(file_path)}")

    def update_ratio(self, *args):
        """Updates the plot aspect ratio."""
        self.generate_fractal()
        self.update_status(f'Plot ratio updated to {self.ratio_var.get()}')

    def on_select(self, eclick, erelease):
        """Handles the selection of a rectangle area."""
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        self.center_x = (x_min + x_max) / 2
        self.center_y = (y_min + y_max) / 2
        self.eps = max(x_max - x_min, y_max - y_min) / 2
        self.update_status(f'Zoomed in to x={self.center_x:.14f}, y={self.center_y:.14f}, eps={self.eps:.14f}')
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, f"{self.center_x:.14f}")
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, f"{self.center_y:.14f}")
        self.eps_entry.delete(0, tk.END)
        self.eps_entry.insert(0, f"{self.eps:.14f}")
        self.generate_fractal()

    def update_status(self, message):
        """Updates the status bar message."""
        self.status_var.set(message)
        self.status_bar.update_idletasks()

    def on_exit(self):
        """종료시 호출될 함수"""
        plt.close('all')  # 모든 matplotlib 플롯 닫기
        self.root.quit()  # tkinter 메인 루프 종료
        self.root.destroy()  # tkinter 윈도우 파괴

# 테트레이션 프랙탈 탐색기 시작
if __name__ == "__main__":
    TetrationFractalExplorer()
