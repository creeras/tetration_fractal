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

        # 객체 설정, 초기화에서 빼고 함수파트에만 넣으려고 했는데, 에러나서 다시 추가.
        self.fig, self.ax = plt.subplots()   # figsize를 설정하지 않음
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 메뉴 바 설정
        self.menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_exit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # 테트레이션 함수 선택 메뉴 추가
        self.tetration_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.tetration_functions = {
            "Normal Tetration": self.normal_tetration,
            "Divergent Tetration": self.divergent_tetration,
            "ln & Exp combi": self.ln_exp_combi
        }
        self.selected_tetration_function = tk.StringVar(value="Normal Tetration")
        for name in self.tetration_functions:
            self.tetration_menu.add_radiobutton(label=name, variable=self.selected_tetration_function, command=self.update_fractal)
        self.menu_bar.add_cascade(label="Tetration Function", menu=self.tetration_menu)

        self.tetration_function = self.normal_tetration # init tetration function

        # 페이즈(magnitude, angle) 선택 메뉴 추가
        self.phase_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.phase_options = {
            "Magnitude(R)": self.fractal_magnitude,
            "Angle(θ)": self.fractal_angle
        }
        self.selected_phase_value = tk.StringVar(value="Magnitude(R)")
        for name in self.phase_options:
            self.phase_menu.add_radiobutton(label=name, variable=self.selected_phase_value, command=self.update_fractal)
        self.menu_bar.add_cascade(label="Phase", menu=self.phase_menu)

        # cmap 리스트 선택 메뉴 추가
        self.cmap_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.cmap_options = ["hsv", "viridis", "plasma", "inferno", "magma", "gray"]
        self.selected_cmap_value = tk.StringVar(value="hsv")
        for cmap in self.cmap_options:
            self.cmap_menu.add_radiobutton(label=cmap, variable=self.selected_cmap_value, command=self.update_colormap)
        self.menu_bar.add_cascade(label="Colormap", menu=self.cmap_menu)

        # 이건 따로인 듯
        self.root.config(menu=self.menu_bar)

        # 상태 바 설정
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")

        # 초기 프랙탈 좌표 경계 설정
        self.center_x, self.center_y = 4, 0
        self.eps = 9

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
        self.max_iter_options = [0, 1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200, 500]
        self.max_iter_var = tk.StringVar(value=str(self.max_iter_options[3]))

        # 콤보박스 위젯 설정
        self.max_iter_dropdown = ttk.Combobox(self.settings_frame, textvariable=self.max_iter_var, value=self.max_iter_options)
        self.max_iter_dropdown.current(3)
        self.max_iter_dropdown.bind("<<ComboboxSelected>>", self.update_fractal)

        # 콤보박스 패킹
        self.max_iter_dropdown.pack(side=tk.LEFT, padx=10)

        # 플롯 화면 비율 옵션 설정
        ttk.Label(self.settings_frame, text="Resolution:").pack(side=tk.LEFT, padx=10)
        self.ratio_dropdown = ttk.OptionMenu(self.settings_frame, self.ratio_var, self.ratio_options[2], *self.ratio_options, command=self.update_ratio)
        self.ratio_dropdown.pack(side=tk.LEFT, padx=10)

        # 좌표 및 eps 값 입력 설정
        ttk.Label(self.settings_frame, text="x=").pack(side=tk.LEFT, padx=5)
        self.x_entry = ttk.Entry(self.settings_frame, width=20)
        self.x_entry.pack(side=tk.LEFT, padx=5)
        self.x_entry.insert(0, f"{self.center_x:.14f}")

        ttk.Label(self.settings_frame, text="y=").pack(side=tk.LEFT, padx=5)
        self.y_entry = ttk.Entry(self.settings_frame, width=20)
        self.y_entry.pack(side=tk.LEFT, padx=5)
        self.y_entry.insert(0, f"{self.center_y:.14f}")

        ttk.Label(self.settings_frame, text="eps=").pack(side=tk.LEFT, padx=5)
        self.eps_entry = ttk.Entry(self.settings_frame, width=20)
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

    def normal_tetration(self, z, max_iter):
        """Computes the fractal using simple exponential iteration."""
        result = cp.copy(z)
        for _ in range(max_iter):
            try:
                cp.power(z, result, out=result)  # 메모리 재할당 없이 직접 수정.
            except OverflowError:
                result = cp.inf
        return result
    
    def divergent_tetration(self, z, max_iter):
        """Computes the fractal using simple exponential iteration."""
        print(f'Function : Divergent_tetration begins / New Version')
        escape_radius = 1e+10 

        result = cp.copy(z)
        divergence_map = cp.zeros(z.shape, dtype=cp.bool_)

        for _ in range(max_iter):
            cp.power(z, result, out=result)  # 메모리 재할당 없이 직접 수정.
            mask = cp.abs(result) > escape_radius
            divergence_map[mask] = True
            z[mask] = cp.nan # 발산한 지점은 더 이상 계산하지 않음.
        # print(f'발산(True) 평균값 = {cp.mean(divergence_map)}')
        divergence_map = 1-divergence_map # 흑/백 반전용
        return divergence_map

    def ln_exp_combi(self, z, max_iter):
        """
        result = 반복:(e^(result*ln(z))
        """
        result = cp.zeros(z.shape, dtype=cp.complex128) + 1 # 초기값을 1로 세팅할 때 +1
        for _ in range(max_iter):
            try:
                result = cp.exp(result * cp.log(z))
            except OverflowError:
                result = cp.inf
        return result
    
    def generate_fractal(self):
        """
        Generates and displays the fractal. 
        플롯 이미지 생성
        """
        # print(f"Starting generate_fractal(self)")
        self.update_status("Generating fractal...")

        self.regenerate_canvas()
        self.max_iter = int(self.max_iter_var.get())
        Z, self.x_range, self.y_range = self.calculate_plot_range(self.max_iter)

        # 현재 선택된 테트레이션 함수 및 페이즈(r,θ) 선택 적용
        self.tetration_function = self.tetration_functions[self.selected_tetration_function.get()]
        self.fractal = self.tetration_function(Z, self.max_iter)
        self.phase_func = self.phase_options[self.selected_phase_value.get()]
        
        print(f'selected_tetration_function: {self.selected_tetration_function.get()}')
        print(f'selected_phase:  {self.selected_phase_value.get()}')

        self.plot_fractal(self.max_iter, self.x_range, self.y_range, self.phase_func) # plot_fractal_alter 를 사용해 볼 수도 있으나... 별로인듯?

    # phase_func() 함수 정의
    def fractal_magnitude(self, fractal):
        return cp.log1p(cp.abs(fractal)) # fractal 함수 결과가 양측으로 극단적으로 나뉘는 상황임 cp.log 또는 cp.log1p 를 추가함.
    def fractal_angle(self, fractal):
        return cp.angle(fractal)
    
    def regenerate_canvas(self):
        """
        self.fig, self.ax, self.canvas 객체 삭제 후 재생성
        원래는 불필요한 과정이 맞으나, 
        이상하게 컬러바가 삭제되지 않고 추가 생성되며 플롯 영역이 줄어드는 문제가 있어서 
        캔버스를 삭제하고 새로 그리기로 함. 
        스트레스 받아서 일단 이대로 쓸 것임. 
        """

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
        print(f'Function : calculate_plot_range begins / New Version')
        print(f'max_iter : {max_iter}')
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

        # 플롯용 x와 y의 값 설정
        # x = self.center_x
        # y = self.center_y
        x = np.float64(self.center_x)
        y = np.float64(self.center_y)
        eps = self.eps
        eps_y = self.eps / self.aspect_ratio
        xlim = (x - eps, x + eps)
        ylim = (y - eps_y, y + eps_y)

        print(f'eps: {self.eps}, eps_y: {eps_y}')
        print(f'xlim: {xlim}, ylim: {ylim}')

        # 중심점&eps로 플롯할 사각형 범위(range)를 얻음. cp np 고민
        x_range = np.linspace(xlim[0], xlim[1], self.plot_width)
        y_range = np.linspace(ylim[0], ylim[1], self.plot_height)

        print(f'x_range(max: {np.max(x_range)}, min: {np.min(x_range)})')
        print(f'y_range(max: {np.max(y_range)}, min: {np.min(y_range)})')

        X, Y = np.meshgrid(x_range, y_range)
        Z = X + 1j * Y

        # 좌표 간격이 큰 경우 데이터 타입을 축소해도 양 옆 픽셀간 구분이 가능함.
        # 좌표 간격이 작으면 작을 수록 더 정밀한 값으로 계산해야 옆 픽셀간 구분이 가능해짐.
        #  1080p 기준, 7e-4 부터 슬슬 조짐이 보이며 1e-4일 때는 일 때 많이 거슬리네요. 
        threshold_complexity = 7e+3 # 7e-3

        if eps < threshold_complexity * (self.plot_width/1920):
            Z = Z.astype(cp.complex128)
        else:
            Z = Z.astype(cp.complex64)

        Z = cp.array(Z)  # GPU 연산을 위해 CuPy 배열로 변환
        print(f'Z.shape: {Z.shape}')
        print(f'Z[0,0]: {Z[0,0]}')
        return Z, x_range, y_range # 함수 분리로 반환 필요. 

    def plot_fractal(self, max_iter, x_range, y_range, phase_func):
        if self.eps > 0:
            self.ax.clear()
            extent = [x_range.min(), x_range.max(), y_range.min(), y_range.max()]

            # divergent는 true/fase 이므로 magnitede/angle 에 따른 처리가 필요없음.
            if self.tetration_function == self.divergent_tetration: 
                fractal = cp.asnumpy(self.fractal)
            else:
                fractal = cp.asnumpy(phase_func(self.fractal))

            cmap = self.selected_cmap_value.get()  # 선택된 컬러맵
            self.ax.imshow(fractal, extent=extent, cmap=cmap, origin='lower')
            self.ax.set_xlabel("Re")
            self.ax.set_ylabel("Im")
            self.ax.set_title(f'Tetration Fractal: Iterations={max_iter}\n'
                            f'x={self.center_x}, y={self.center_y}, eps={self.eps}')
            # self.ax.set_autoscale_on(False)

            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.fig.colorbar(self.ax.get_images()[0], ax=self.ax)
            self.canvas.draw()
            print('- '*30)
            self.update_status("Fractal generated. Ready...")
        else:
            self.update_status("Warning: eps must be positive")


    def update_colormap(self, *args):
        """
        colormap 변경시 호출됨
        """
        self.regenerate_canvas() # 원래는 이것도 필요없어야 맞는건데, 칼라바 무한증식 현상이 나타나서 어쩔 수 없음.
        self.plot_fractal(self.max_iter, self.x_range, self.y_range, self.phase_func) 
        print(f'update_colormap: {self.selected_cmap_value.get()}')


    def update_fractal(self, *args):
        """Updates the fractal based on maximum iterations.
        max_iter 리스트에서 선택시, 
        메뉴에서 함수 선택시
        호출됨
        """
        # print(f'args: {args}')
        function_name = self.selected_tetration_function.get()
        self.tetration_function = self.tetration_functions[function_name]

        # print(f'args: {args}')
        phase_name = self.selected_phase_value.get()
        self.phase_menu = self.phase_options[phase_name]

        try:
            self.max_iter = int(self.max_iter_var.get())

            #if args:
            #    self.max_iter_var.set(args[0])  # max_iter에서 호출할 때만
            self.generate_fractal()
            self.update_status(f'Updated max iterations to {self.max_iter_var.get()}')
        except ValueError:
            self.max_iter_var.set(str(self.max_iter_options[3]))
            self.max_iter = self.max_iter_options[3]

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
        print(f'apply_coordinates: {self.center_x}, {self.center_y}, {self.eps}')

    def zoom(self, *args):
        """Zooms in or out based on the selected zoom factor."""
        zoom_factor_str = self.zoom_var.get()
        zoom_factor = float(zoom_factor_str[1:]) if "x" in zoom_factor_str else 1 / float(zoom_factor_str[2:])
        self.eps /= zoom_factor
        self.eps_entry.delete(0, tk.END)
        self.eps_entry.insert(0, f"{self.eps:.14f}")
        self.generate_fractal()
        self.update_status(f'Zoomed {"in" if zoom_factor > 1 else "out"} by {zoom_factor}')
        print(f"Zoom: {zoom_factor}")

    def reset(self):
        """Resets the fractal to the initial state."""
        self.center_x, self.center_y = -0.5, 0
        self.eps = 1
        self.max_iter_var.set("100")
        self.generate_fractal()
        self.update_status("Reset to initial state")
        print(f'resetted')

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

            # 플롯용 x와 y의 값 설정
            x = self.center_x
            y = self.center_y
            eps = self.eps
            eps_y = self.eps / self.aspect_ratio
            xlim = (x - eps, x + eps)
            ylim = (y - eps_y, y + eps_y)
            
            # self.fig.set_size_inches(self.plot_width / 100, self.plot_height / 100)  # 100 DPI로 가정
            # 이게 있어야 맞는 건지 없어야 맞는 건지?

            x_range = np.linspace(xlim[0], xlim[1], self.plot_width)
            y_range = np.linspace(ylim[0], ylim[1], self.plot_height)

            extent = [x_range.min(), x_range.max(), y_range.min(), y_range.max()]

            phase_func = self.phase_options[self.selected_phase_value.get()]

            # divergent는 true/fase 이므로 magnitede/angle 에 따른 처리가 필요없음.
            if self.tetration_function == self.divergent_tetration: 
                fractal = cp.asnumpy(self.fractal)
            else:
                fractal = cp.asnumpy(phase_func(self.fractal))

            cmap = self.selected_cmap_value.get()  # 선택된 컬러맵
            ax.imshow(fractal, extent=extent, cmap=cmap, origin='lower')

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
        print(f'width: {self.plot_width}, height: {self.plot_height}')

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
        print(f'center: {self.center_x}, {self.center_y}, eps={self.eps} selected')

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
