## AI_Parking_Occupancy_Detection(Segmentation)
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e88edaf2-1b40-47f6-a3fb-821d46000a67)
 
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


# 1. 데이터셋 분석 및 클래스 EDA

1. 데이터 소개 
    - AI-Hub (주차 공간 탐색을 위한 차량 관점 복합 데이터)
        
       ![image](https://github.com/user-attachments/assets/e990b950-ddf0-4276-a074-cce91caf2f27)

        
       ![image (1)](https://github.com/user-attachments/assets/57479ba2-47bb-4be4-b474-a47e79a14b40)

        
        : 복합 데이터 목록 중 1️⃣**수량이 가장 많은** 데이터셋 뿐만 아니라,  2️⃣**LiDAR 로 촬영한 포인트 클라우드 데이터 파일**과 3️⃣**PCD 파일 포맷**으로 수집된 이미지 데이터가 가장 많았다. 그 중 LiDAR 데이터는 370번 폴더부터 443폴더까지 존재한다.
   
# 2. 클래스 정보와 EDA

1. bbox3d와 segmentation의 클래스 정보 
    
    
    | Category | Count | Source |
    | --- | --- | --- |
    | Drivable Space | 36,191 | Segmentation |
    | Parking Space | 40,374 | Segmentation |
    | Car | 261,966 | bbox3d |
    | Cycle | 4,524 | bbox3d |
    | Pedestrian | 554 | bbox3d |
    
    ![image (2)](https://github.com/user-attachments/assets/1da0eb6f-a11c-44e3-a69d-611be63b29fd)
    
2. bbox3d 투영 결과
    - bbox3d를 LiDAR의 pcd에 투영
        
        ![image (3)](https://github.com/user-attachments/assets/7de7eb14-b6a4-4b20-bda0-64be6e2fbd17)

        
    - 3차원 투영 결과를 2차원 평면에 나타내기
        
       ![image (4)](https://github.com/user-attachments/assets/8568cfa8-bce2-4681-abd4-80d6805087c1)

        
    - ‘point cloud data’ 읽고 ‘bbox3d’ 데이터 이용해서 3D 포인트 클라우드 시각화하기
        
       [영상]
        
    - bbox3d 이미지에 투영
        
       ![image (5)](https://github.com/user-attachments/assets/3faa3ec0-c686-402d-a3cc-dee681dbd202)
       원래 하려고 했던 것
      



        ![download](https://github.com/user-attachments/assets/01194e02-cfae-4098-980d-a1ed3b5c4ae9)
      나온 결과 
      
        ⇒ 투영된 결과를 살펴본 끝에, 구체적인 프로젝트 주제를 도출하기 어려운 것으로 판단하였다. 이에 따라, 기존에 진행하려고 했던 Point Cloud 기반의 작업은 중단하기로 결정했다.
        
3. Segmentation 투영 결과
    - ‘Segmentation’ 의 폴리곤 좌표를 이미지에 투영하기
        
        ![image (6)](https://github.com/user-attachments/assets/935918cd-6b46-4945-8dcb-8f61fba3364f)

    - ‘Segmentation’ + ‘bbox2d’좌표를 이미지에 투영하기
        
       ![image (7)](https://github.com/user-attachments/assets/7ca5a313-a9ae-4ab1-b7f3-2e4511e4a656)

    

# 3. 데이터셋 전처리 과정

1. JSON 파일에서 Segmentation 정보 중 "Driveable Space"와 "Parking Space" 클래스에 해당하는 데이터만 선별하였다.
2. 필터링된 JSON 파일은 총 30,000개 이상이었으나, 랜덤 샘플링을 통해 약 8,000개로 데이터를 축소하였다.
3. 초기에는 16,000개의 이미지를 6:2:2 비율로 훈련, 검증, 테스트 데이터셋으로 분할하여 모델을 학습시키고자 했으나, OOM(Out of Memory) 문제가 발생하였다. 이를 해결하기 위해 데이터 크기를 절반으로 줄였으며, 이미지와 JSON 파일의 라벨 데이터를 1/3 크기(640x360)로 리사이즈하여 학습을 진행했다.
    
    ![image (8)](https://github.com/user-attachments/assets/fe671994-fb5a-4d57-9e0e-bee009739ad4)

    최종적으로 사용된 데이터셋 수
    

# 4. 모델 학습

## 1. U-Net

- 1차 U-Net : `lr=1e-3, batchs=4, epochs=5`
    - 검증
        
       ![image (9)](https://github.com/user-attachments/assets/99c08a41-f5a8-4735-9bee-36fbd77b5366)

    - 결과 시각화
        
        `Overall Mean IoU: 0.6662`
        ![image (10)](https://github.com/user-attachments/assets/083978cf-5102-4ff8-b35f-7dfac35e3643)

        
        - 성능 개선 방안
            
            : 라벨 값에 대한 예측 결과는 전반적으로 나쁘지 않았으나, 손실 함수 계산 과정에서 문제가 발생한 것으로 보인다. 특히, Loss 값이 음수로 출력되었으며, 이는 손실 함수 계산 방식이나 관련 설정에 오류가 있음을 의미한다.
            
- 2차 U-Net :  `lr=1e-4, batchs=4, epochs=5`
    - validation loss 해결을 위한 변경사항
        - 손실함수 변경 `BCEWithLogitsLoss` → `CrossEntropyLoss` (다중클래스 문제)
        - 학습률 감소시켜보기 `1e-3` → `1e-4`
        - 라벨 텐서 차원 조정 `labels = labels.squeeze(1).long()  #(N, 1, H, W) -> (N, H, W)`
        - 모델 출력함수 변경 `torch.sigmoid(outputs)` → `torch.softmax(outputs, dim=1)` (다중클래스 문제)
    - 검증
        
        `Overall Mean IoU: 0.8042`
        
        ![image (11)](https://github.com/user-attachments/assets/67a96344-50e2-490f-845d-24ced114d561)

    - 결과 시각화
        
        ![image (12)](https://github.com/user-attachments/assets/b7aff5a5-634e-4bae-ae99-42458d5b2c00)

- 3차 U-Net :  `lr=1e-4, batchs=4, epochs=50`
    - 검증
        ![image (13)](https://github.com/user-attachments/assets/0473e513-5c18-4669-ae47-c8db1ba3971e)

        과적합이 발생하려고 했던 모습이 보인다. 
        ⇒ 사용한 데이터셋의 특성상 유사한 이미지로 구성되어 있기 때문으로 추정된다.
        
        `Overall Mean IoU: 0.8710`
        
    - 결과 시각화
        
       ![image (14)](https://github.com/user-attachments/assets/99cbc5e8-d660-4cce-b060-0ec5768ced90)


## 2. Mask-RCNN

- 1차 Mask-RCNN
    - 최적화 Config : 최적화 모델로 Adam 을 사용하였는데, Mask-RCNN **모델 구조에 따라** learning rates 를 다르게 설정하였다.
        - **backbone** : `lr=0.00001`
        - **RPN** : `lr = 0.0001`
        - **ROI** (region of interest) : `lr = 0.0001`
        
        ---
        
    - 모델 학습 Config : `epoch = 5, lr = 0.0001, batch = 8`
    - 검증
        
       ![{4C8197F1-DDA6-42D5-BA31-22B0AD9B576F}](https://github.com/user-attachments/assets/84134341-2c4b-4291-be17-a07558d9bfd2)

        
        ⇒ 모델이 지정 데이터를 학습함에 있어 에폭수가 조금씩 높아짐에 따라 **Train loss** 와 **Validation loss** 모두 감소하는 경향을 보인다. 특히 **Train loss** 는 상당한 감소율을 보여준다.
        
        ⇒ 의문인 점은 에폭 3 → 에폭 4 로 학습시키는 과정에서 **Validation loss**가 증가했다는 점.
        
        ⇒ 에폭수=50 /  동일한 lr & batch size 로 2차 학습을 시킨 이후에도 유사한 결과값이 나온다면 learning rate 를 조금 감소시켜 다시 학습시킬 예정이다.
        
    - 결과 시각화
        
       ![maskrcnn4](https://github.com/user-attachments/assets/702b7381-d62e-473f-bb98-faf8452dfd49)

        - Detection Statistics :
            - `Driveable Space` : **1 objects, Average confidence: 0.997**
        
        ![maskrcnn5](https://github.com/user-attachments/assets/3cfb4c17-4269-481e-8b32-23fd801e5a7a)

        - Detection Statistics
            - `Parking Space` : **1 objects, Average confidence: 0.980**
            - `Driveable Space` : **1 objects, Average confidence: 0.998**
        
        ![maskrcnn6](https://github.com/user-attachments/assets/27e5932d-6a98-409e-b33e-258e30548c81)

        - Detection Statistics
            - `Parking Space` : **1 objects, Average confidence: 0.909**
            - `Driveable Space` :  **2 objects, Average confidence: 0.997**
                
                → 결과가 말도 안되게 좋다. 데이터를 다 외워버린듯하다 → 과적합 예상
                
        
        ---
        
- 2차 Mask-RCNN  💀 A100으로 50 에폭돌리는데 지금 두시간 30분 동안 돌리는 중…
    - Config : `epoch = 50, lr = 0.0001 , batch = 8`
    - 검증
        
       ![스크린샷 2024-11-20 155107](https://github.com/user-attachments/assets/3287415e-d6cc-4764-948c-1f310d5a03a7)

       ![스크린샷 2024-11-20 161350](https://github.com/user-attachments/assets/4db356c5-42d3-4789-ae1e-a0ebe716db18)

        
        ⇒ 과적합입니다
        
    
    - 결과 시각화
        
        ![epoch50](https://github.com/user-attachments/assets/59897669-94c6-4c31-8919-3a86ccd42518)

       ![epoch50_2](https://github.com/user-attachments/assets/e8b7f42f-580d-4cd0-8a76-31c33a21f066)

      ![epoch50_3](https://github.com/user-attachments/assets/7f0d7fcf-50b2-45a5-81a3-01d026fbc473)

---

## 3. DeepLabv3+

**1차 DeepLabv3+**

- `epochs = 5, lr = 0.0001, batch=4`
- 결과 시각화

![download (1)](https://github.com/user-attachments/assets/8c130252-3e1d-4463-a0a5-61e803abbf4b)
![download (2)](https://github.com/user-attachments/assets/d741c34a-1d2a-44ea-9041-d3160e81df19)

`Test Loss: 0.0349`
![download (3)](https://github.com/user-attachments/assets/8cbce063-424b-405a-b716-548166e0ffaf)

![download (4)](https://github.com/user-attachments/assets/5c4d5ad4-2fbc-492d-a877-5e9288f640d5)
![download (5)](https://github.com/user-attachments/assets/bc6bad5b-c6a9-4a45-b84b-98e5ed38595a)

`Train Loss: 0.0249, Validation Loss: 0.0344`

낮은 에폭임에도 U-Net보다 좋은 성능을 보였다

**2차 DeepLabv3+**

- `epochs = 50, lr = 0.0001, batch = 4`
- 결과 시각화

![download (6)](https://github.com/user-attachments/assets/c4d87a0d-8098-4cb1-87d9-de5c8c557a5f)

![download](https://github.com/user-attachments/assets/f5b5b3d4-46a2-4b7f-8440-ed9fbe4e80db)

![download (18)](https://github.com/user-attachments/assets/f499431e-ad9e-4d03-9aa9-4913386f1d6b)

`Test Loss: 0.0422`

![download](https://github.com/user-attachments/assets/c84ad0f9-ad9e-4082-87e0-83ab7a791492)

![download (1)](https://github.com/user-attachments/assets/114f6a90-cbce-4c86-9535-6bbb2bb47eae)
![download (2)](https://github.com/user-attachments/assets/72e57437-0cce-4b11-8d16-28ca49ba42b9)

보다 개선된 세그멘테이션 결과를 확인할 수 있었다

Train Loss: 0.0077, Validation Loss가 0.0418로 안정적인 수준을 보였다.
![image](https://github.com/user-attachments/assets/52e14e98-868d-4909-9dc9-134ee15236a0)


`mIoU: 0.8717` → 준수한 결과를 보임

**3차 DeepLabv3**

- `epochs = 100, lr: = 0.001, batch = 4`
- 결과 시각화
![download (3)](https://github.com/user-attachments/assets/6040dd6e-7775-4027-983d-7e8a41f433d5)

![download (4)](https://github.com/user-attachments/assets/a7a7c65d-76a9-4228-bb2d-cc2762dd54ec)
![download (5)](https://github.com/user-attachments/assets/cf0893c3-e192-4232-98be-0d983c15ffd5)

`Test Loss: 0.0688`

![download (6)](https://github.com/user-attachments/assets/151c1e27-7abe-41cb-b731-e8bf75e0b503)
![download (7)](https://github.com/user-attachments/assets/4f87f8c7-1cb1-4787-b0a5-5737347c0847)
![download (8)](https://github.com/user-attachments/assets/d467d615-2cec-4731-ac52-4f1e2fc68bc9)

더 세밀하게 세그멘테이션을 수행한 모습을 확인할 수 있었다.

`Train Loss는 0.0056, Validation Loss는 0.0629`로 다소 높은 수치를 기록
![image (1)](https://github.com/user-attachments/assets/a09bd64b-f221-4864-a229-7163eaf81c66)


`mIoU: 0.8353` → 감소하였다

학습률을 높인 결과 손실이 증가하여 과적합의 징후를 보여 추가적인 튜닝이 필요함을 보였다.

---

# 4. U-NET vs. Mask-RCNN vs. DeepLabV3+

## 1. 모든 모델 학습 및 평가 후 Segmentation 예측 결과 비교

### (1) 학습 및 평가 실행

![{75148D8A-3A9C-4C58-9071-C5DAD1347950}](https://github.com/user-attachments/assets/460743e0-83b6-4805-964a-6b05b834190f)

⇒ 세가지 모델을 비교한 결과, 
Mask R-CNN >> DeepLabv3+ >> U-Net 순서로 세그멘테이션 감지가 잘 되는 것으로 보인다.

### (2) 각 모델 세그멘테이션 결과 시각화

- 이미지 예시 1

![모든모델비교1](https://github.com/user-attachments/assets/75f7c2d3-d956-4601-88f7-6c672f09a6d8)

- 이미지 예시 2

![모든모델비교2](https://github.com/user-attachments/assets/307b8c34-5107-44f7-b5d3-5917750b706d)

- 이미지 예시 3
![모든모델비교3](https://github.com/user-attachments/assets/d5cf207e-1405-4008-9517-1f799a7e9cef)


## 시연영상
(DeepLabV3+)
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e88edaf2-1b40-47f6-a3fb-821d46000a67)

(U-Net)
![segmentation_imtovi_output-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/95a33910-a999-4a16-b24f-d36ecdd529d5)

(Mask R-CNN)
![Y2meta app-parkingspace_video_output_maskrcnn-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/9962bee8-c208-4364-8591-b293a0074c2e)

