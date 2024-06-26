1. Tetration 계산 (GPU_CUDA)
   - GPU 버전은 파이선 cupy 라이브러리를 이용하여 테트레이션 계산을 수행합니다.
   - 기본 계산식 = cp.exp(result * cp.log(z))
     - np.log(z) : z의 자연로그를 취합니다. 지수함수에 의한 급격한 결과변화를 제어하여, 안정성을 높이기 위함입니다.
     - np.exp(arg) : e^arg를 반환합니다. 지수(arg)가 복소수인 특징이 있습니다.
   - 데이터타입
     - dtype=cp.complex128 : 가장 깊은 곳까지 줌인하기 위해서는 complex128 로 해야 함.
     - 하지만 cp.complex64 를 사용할 경우 계산 속도가 빨라지므로, 큰값에서는 complex64 를 쓰는 편이 좋음. (간단한 조건문으로 처리 가능)
   - 반복횟수
     - 반복횟수가 많을수록 결과 이미지가 개선되는 효과가 있음. 수렴/발산을 찾는 것도 그렇고 이 경우처럼 컬러 이미지를 얻는 경우에도 마찬가지.
     - 다만 반복 횟수는 계산 시간에 정비례하는 관계이므로 적당한 수준을 찾는 것이 중요함. 
     - gpu 버전은 속도가 상대적으로 빠르기 때문에 기본값 100회로 설정하였습니다. self.max_iter_options[3]
   - 기본 해상도 
     - 대부분의 모니터가 FHD(1080p) 인 것은 맞으나, 프로그램 특성상 플롯 이미지크기는 화면을 꽉 채울 수 없어 프로그램으로만 플롯을 볼 때는 1080p가 시간낭비일 수 있습니다. 
     - 그래서 화면 표시 용도로는 1080p를 720p로 변경하였습니다. 

2. CPU 버전과 차이
   - CPU 버전은 계산 속도가 느려 이것저것 적용해가며 공부하기에는 여러모로 어렵습니다.
   - 그래서 GPU 버전으로 프로그램을 개선해 나가며 공부해 갈 계획이며, GPU 버전의 내용이 CPU 버전에 모두 적용되지는 않을 것입니다. 

3. trace_coordinate_divergence.py
   - 시작좌표&eps1, 끝좌표&eps2, 생성할 프레임 수를 입력하면 연속적인 sequence 이미지를 png 파일로 저장합니다.
   - 1920*1080 해상도일 때, 평균 100~200 KB/장 정도로 만들어지는 것 같습니다. (아주 빈땅일 때는 수KB 입니다) 
   - 현재는 연속 데이터라고 해서 이미지 생성 속도에 이득이 있는 것은 아닙니다.
   - 이론적으로는 이전 이미지와 겹치는 부분은 데이터를 재활용한다거나 할 수도 있겠으나, 재활용도 무제한 가능한 건 아닐테고, 몇번에 한번씩은 제대로 된 이미지를 만들긴 해야 할 겁니다.
   - 그보다도... 이미지 생성 보간 기술을 쓰는 것이 훨씬 좋을 듯 합니다.
   - 데이터 연속성이 크고, 중간에 튀는 이미지가 생성되는 일이 없기 때문에 아주 유용할 것으로 판단되고, 요즘은 AI 기술도 아주 좋기 때문에 충분히 가능할 것으로 보입니다.
