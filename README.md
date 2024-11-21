## AI_Parking_Occupancy_Detection(Segmentation)
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e88edaf2-1b40-47f6-a3fb-821d46000a67)
- 최종 영상 (U-Net)
  [![Video Label](http://img.youtube.com/vi/B96UbpyoJ6A/0.jpg)]( https://youtu.be/B96UbpyoJ6A)
 


## 프로젝트 소개
- 다양한 Segmentation 모델(U-Net, Mask-RCNN, DeepLabv3+)을 사용해서 자율 주행 자동차의 실내 주차 환경에서 주행가능 영역 및 주차 공간 탐지 성능을 비교하고, 각 모델의 성능을 최적화합니다.


## 사용 데이터셋(Dataset)
AI-Hub의 '주차 공간 탐색을 위한 차량 관점 복합 데이터'의 실내중형주차장 데이터를 선별해서 사용했습니다.
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=598

그 중에서 클래스 정보가 Segmenatation('Parking Space', 'Drivable Space')에 해당하는 부분만 필터링했습니다.


## 사용 모델 소개
##### 1. U-Net 
: 의료 영상 세그멘테이션에 주로 사용되던 구조로, 주차 공간 감지에서 픽셀 단위의 정확도를 높입니다.
##### 2. Mask-RCNN 
: 객체 탐지와 세그멘테이션을 동시에 수행하는 모델로, 주차 공간 및 주행 공간을 탐지라는데 효과적입니다.
##### 3. DeepLabv3+ 
: 심층 네크워크와 공간 피라미드 풀링을 결합하여 복잡한 장면에서도 높은 정확도를 제공합니다.





## Prerequisites
- **Python 3.10** 🐍
- The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:
    ```bash
    pip install -r requirements.txt
    ```

## Contributors ✨
<table>
    <tbody>
        <tr>
            <td align="center" valign="top" width="14.28%"><a href="https://github.com/rwambangho"><img src="https://avatars.githubusercontent.com/u/121777977?v=4" width="100px;" alt="ByeongHo Yoon"><br/><sub><b>ByeongHo Yoon</b></ub><a><br/>
            <td align="center" valign="top" width="14.28%"><a href="https://github.com/JaeHeeLE"><img src="https://avatars.githubusercontent.com/u/153152453?v=4" width="100px;" alt="Min Joo Lee"><br/><sub><b>JaeHee Lee</b></ub><a><br/>
            <td align="center" valign="top" width="14.28%"><a href="https://github.com/sinya3558"><img src="https://avatars.githubusercontent.com/u/70243358?v=4" width="100px;" alt="Min Joo Lee"><br/><sub><b>Seunga Kim</b></ub><a><br/>
        </tr>
    </tbody>
</table>
