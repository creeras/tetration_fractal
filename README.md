# tetration_fractal
테트레이션 계산에 의해 생기는 프렉탈 구조를 
이미지화하는 과정을 공부해 보려고 만듦.

(제가 처음 본) 오리지널 프로젝트는 이 곳입니다. 
https://github.com/DMTPARK/mytetration

원래는 테트레이션 -> 발산여부만 파악 -> 플롯 이미지화 하는 것입니다. 
```
               for k in range(max_iter):
                z = c_val ** z
                if np.abs(z) > escape_radius:
                    divergence_map[i, j] = True
                    break
```
흑백의 프렉탈 이미지도 충분히 환상적이긴 합니다만, 
테트레이션 계산 과정/결과를 다양하게 바꾸어 보면서 
다양하고 화려한 무늬를 살펴볼 수도 있지 않을까? 하는 생각에 
프로그램을 확장 제작하게 되었습니다. 

의견을 주셔도 좋고, 개선을 해 주셔도 좋습니다. 
