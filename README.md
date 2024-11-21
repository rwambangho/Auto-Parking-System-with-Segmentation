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
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/a0fe9294-e33c-4257-a049-feeb2d5995d6/image.png)
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/b15f4081-d259-40b0-b3ab-6bc13336fd58/image.png)
        
        : 복합 데이터 목록 중 1️⃣**수량이 가장 많은** 데이터셋 뿐만 아니라,  2️⃣**LiDAR 로 촬영한 포인트 클라우드 데이터 파일**과 3️⃣**PCD 파일 포맷**으로 수집된 이미지 데이터가 가장 많았다. 그 중 LiDAR 데이터는 370번 폴더부터 443폴더까지 존재한다.
        
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/e8a56648-a3b7-460b-9fbd-086014c1f848/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/3b030004-07d3-4794-adfc-5e8a6fbd1a21/image.png)
    

# 2. 클래스 정보와 EDA

1. bbox3d와 segmentation의 클래스 정보 
    
    
    | Category | Count | Source |
    | --- | --- | --- |
    | Drivable Space | 36,191 | Segmentation |
    | Parking Space | 40,374 | Segmentation |
    | Car | 261,966 | bbox3d |
    | Cycle | 4,524 | bbox3d |
    | Pedestrian | 554 | bbox3d |
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/3c1724ba-3805-4750-ae2e-4887c18d3731/image.png)
    
2. bbox3d 투영 결과
    - bbox3d를 LiDAR의 pcd에 투영
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/304f1fad-c9a7-4866-9d5f-fc7fd5219dbf/bd64436a-09b9-4ff3-84b1-433ddc771de5.png)
        
    - 3차원 투영 결과를 2차원 평면에 나타내기
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/e439d8dd-d48b-4312-adfb-9d641b5565eb/image.png)
        
    - ‘point cloud data’ 읽고 ‘bbox3d’ 데이터 이용해서 3D 포인트 클라우드 시각화하기
        
        [point_cloud_3d시각화.mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/526eaffc-473b-4516-a5c4-b6750b51ce74/point_cloud_3d%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA.mp4)
        
    - bbox3d 이미지에 투영
        
        ![원래 하려고 했던 것](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/c2310fcd-0a1c-4e17-8786-d9c5138f0ef2/image.png)
        
        원래 하려고 했던 것
        
        ![나온 결과](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ebd73c01-fcd5-48ab-b8e0-fce67eacf097/download.png)
        
        나온 결과
        
        ⇒ 투영된 결과를 살펴본 끝에, 구체적인 프로젝트 주제를 도출하기 어려운 것으로 판단하였다. 이에 따라, 기존에 진행하려고 했던 Point Cloud 기반의 작업은 중단하기로 결정했다.
        
3. Segmentation 투영 결과
    - ‘Segmentation’ 의 폴리곤 좌표를 이미지에 투영하기
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/67bcbc4f-9122-4461-98df-4ce0bb4f756f/image.png)
        
    - ‘Segmentation’ + ‘bbox2d’좌표를 이미지에 투영하기
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/5dcc6e2d-871c-463c-93a8-f7b22e706170/image.png)
        
    

# 3. 데이터셋 전처리 과정

1. JSON 파일에서 Segmentation 정보 중 "Driveable Space"와 "Parking Space" 클래스에 해당하는 데이터만 선별하였다.
2. 필터링된 JSON 파일은 총 30,000개 이상이었으나, 랜덤 샘플링을 통해 약 8,000개로 데이터를 축소하였다.
3. 초기에는 16,000개의 이미지를 6:2:2 비율로 훈련, 검증, 테스트 데이터셋으로 분할하여 모델을 학습시키고자 했으나, OOM(Out of Memory) 문제가 발생하였다. 이를 해결하기 위해 데이터 크기를 절반으로 줄였으며, 이미지와 JSON 파일의 라벨 데이터를 1/3 크기(640x360)로 리사이즈하여 학습을 진행했다.
    
    ![최종적으로 사용된 데이터셋 수](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/2337ef67-fff8-469e-a998-1f4eb7b14c74/image.png)
    
    최종적으로 사용된 데이터셋 수
    

# 4. 모델 학습

- 

## 1. U-Net

- 1차 U-Net : lr=1e-3, batchs=4, epochs=5
    - 검증
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/df3d43eb-5473-4552-8e7c-d8759eb2bb57/image.png)
        
    - 결과 시각화
        
        `Overall Mean IoU: 0.6662`
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/d4da64c6-6113-47bc-a54e-b77b61658559/image.png)
        
        - 성능 개선 방안
            
            : 라벨 값에 대한 예측 결과는 전반적으로 나쁘지 않았으나, 손실 함수 계산 과정에서 문제가 발생한 것으로 보인다. 특히, Loss 값이 음수로 출력되었으며, 이는 손실 함수 계산 방식이나 관련 설정에 오류가 있음을 의미한다.
            
- 2차 U-Net :  lr=1e-4, batchs=4, epochs=5
    - validation loss 해결을 위한 변경사항
        - 손실함수 변경 `BCEWithLogitsLoss` → `CrossEntropyLoss` (다중클래스 문제)
        - 학습률 감소시켜보기 `1e-3` → `1e-4`
        - 라벨 텐서 차원 조정 `labels = labels.squeeze(1).long()  #(N, 1, H, W) -> (N, H, W)`
        - 모델 출력함수 변경 `torch.sigmoid(outputs)` → `torch.softmax(outputs, dim=1)` (다중클래스 문제)
    - 검증
        
        `Overall Mean IoU: 0.8042`
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/185f5870-affe-4e83-ac90-86bfdb1c86aa/image.png)
        
    - 결과 시각화
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/6ef636e4-6062-410d-894e-eb606b37900d/image.png)
        
- 3차 U-Net :  lr=1e-4, batchs=4, epochs=50
    - 검증
        
        ![과적합이 발생하려고 했던 모습이 보인다. 
        ⇒ 사용한 데이터셋의 특성상 유사한 이미지로 구성되어 있기 때문으로 추정된다.](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/1c43e88f-769b-4c48-98b2-55a24c0e59f4/image.png)
        
        과적합이 발생하려고 했던 모습이 보인다. 
        ⇒ 사용한 데이터셋의 특성상 유사한 이미지로 구성되어 있기 때문으로 추정된다.
        
        `Overall Mean IoU: 0.8710`
        
    - 결과 시각화
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/92860c0e-3b22-48db-a367-b11d306f270f/image.png)
        

## 2. Mask-RCNN

- 수정이 필요해 보이는 이미지 시각화
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/9ce6bdd9-62bb-4697-8701-8acbe70fc686/image.png)
    
    ⇒ 이미 소형차가 주차한 공간을 주차 가능 공간이라고 감지를 했다. 왜?
    
- 1차 Mask-RCNN
    - 최적화 Config : 최적화 모델로 Adam 을 사용하였는데, Mask-RCNN **모델 구조에 따라** learning rates 를 다르게 설정하였다.
        - **backbone** : `lr=0.00001`
        - **RPN** : `lr = 0.0001`
        - **ROI** (region of interest) : `lr = 0.0001`
        
        ---
        
    - 모델 학습 Config : `epoch = 5, lr = 0.0001, batch = 8`
    - 검증
        
        ![{4C8197F1-DDA6-42D5-BA31-22B0AD9B576F}.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/c2a29a42-c398-4e8d-9c47-ea889c9f31ef/4C8197F1-DDA6-42D5-BA31-22B0AD9B576F.png)
        
        ⇒ 모델이 지정 데이터를 학습함에 있어 에폭수가 조금씩 높아짐에 따라 **Train loss** 와 **Validation loss** 모두 감소하는 경향을 보인다. 특히 **Train loss** 는 상당한 감소율을 보여준다.
        
        ⇒ 의문인 점은 에폭 3 → 에폭 4 로 학습시키는 과정에서 **Validation loss**가 증가했다는 점.
        
        ⇒ 에폭수=50 /  동일한 lr & batch size 로 2차 학습을 시킨 이후에도 유사한 결과값이 나온다면 learning rate 를 조금 감소시켜 다시 학습시킬 예정이다.
        
    - 결과 시각화
        
        ![maskrcnn4.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ff1305d9-97d4-4765-8364-107ef183ecc9/maskrcnn4.png)
        
        - Detection Statistics :
            - `Driveable Space` : **1 objects, Average confidence: 0.997**
        
        ![maskrcnn5.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/e51b9ab9-b2ff-4d2b-81a0-483e03dcca60/maskrcnn5.png)
        
        - Detection Statistics
            - `Parking Space` : **1 objects, Average confidence: 0.980**
            - `Driveable Space` : **1 objects, Average confidence: 0.998**
        
        ![maskrcnn6.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ac297e41-b9c8-4f45-8790-f1976e5efb5c/maskrcnn6.png)
        
        - Detection Statistics
            - `Parking Space` : **1 objects, Average confidence: 0.909**
            - `Driveable Space` :  **2 objects, Average confidence: 0.997**
                
                → 결과가 말도 안되게 좋다. 데이터를 다 외워버린듯하다 → 과적합 예상
                
        
        ---
        
- 2차 Mask-RCNN  💀 A100으로 50 에폭돌리는데 지금 두시간 30분 동안 돌리는 중…
    - Config : `epoch = 50, lr = 0.0001 , batch = 8`
    - 검증
        
        ![스크린샷 2024-11-20 155107.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ce08cd99-15a4-4431-9894-a994161c57d6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2024-11-20_155107.png)
        
        ![스크린샷 2024-11-20 161350.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/857cecf5-7c4d-493a-b060-91b0eed16640/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2024-11-20_161350.png)
        
        ⇒ 과적합입니다
        
    
    - 결과 시각화
        
        
        ![epoch50.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/88373469-f52c-49bc-9096-8b062f73f2aa/epoch50.png)
        
        ![epoch50_2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/0cd1c927-4a3e-4575-9945-68170d2960c6/epoch50_2.png)
        
        ![epoch50_3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/2e970767-efca-4060-83fc-3e50a6c079a4/epoch50_3.png)
        

---

## 3. DeepLabv3+

**1차 DeepLabv3+**

- epochs = 5, lr = 0.0001, batch=4
- 결과 시각화

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ce9cfdfb-8f52-425e-895e-d40b62919cb9/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/83100145-b7c7-4252-b1ba-27d9c9efabfe/download.png)

Test Loss: 0.0349

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/42595255-7f6d-435f-94e2-6083e2a9fa2a/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/7e6cde82-0e57-40c3-b9e9-5a25088596db/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/7355a5ff-9f6a-4ee1-b75d-b708777806bb/download.png)

Train Loss: 0.0249, Validation Loss: 0.0344

낮은 에폭임에도 U-Net보다 좋은 성능을 보였다

**2차 DeepLabv3+**

- epochs = 50, lr = 0.0001, batch = 4
- 결과 시각화

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/4c8ab7f6-5398-4573-bda3-c3b967a00fc9/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/44028368-155a-421a-b1af-ca3cf2f7d7c0/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/ea0ea4cc-6724-4eaf-8b00-a08d999f8e98/download.png)

Test Loss: 0.0422

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/2dfd6c53-d6dc-48b2-88bf-7f9dd7ff1049/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/faa6b20d-e757-49d7-8cee-bc1d9caadf07/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/37f5baeb-f081-4dcd-9e4b-6d2c63caa21d/download.png)

보다 개선된 세그멘테이션 결과를 확인할 수 있었다

Train Loss: 0.0077, Validation Loss가 0.0418로 안정적인 수준을 보였다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/65779f41-7d34-4065-b90b-4165db27248f/image.png)

mIoU: 0.8717 → 준수한 결과를 보임

**3차 DeepLabv3**

- epochs = 100, lr: = 0.001, batch = 4
- 결과 시각화

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/0291ac43-8b28-45e1-933e-a73bb987b37a/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/31fb4ac7-0a35-4399-84de-1d07d3334e00/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/8300da0f-02b7-4e9c-8644-3fb59d08243e/download.png)

Test Loss: 0.0688

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/3729450f-7820-4dca-b4d6-86c3f4630789/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/026254d6-c829-4658-a91a-cb4c8ee7d5fc/download.png)

![download.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/73467d1c-9a28-4775-ad86-c0b16b20e939/download.png)

더 세밀하게 세그멘테이션을 수행한 모습을 확인할 수 있었다.

Train Loss는 0.0056, Validation Loss는 0.0629로 다소 높은 수치를 기록

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/b455e487-b2b1-4eca-8f7d-a9e98c8f0e9a/image.png)

mIoU: 0.8353 → 감소하였다

학습률을 높인 결과 손실이 증가하여 과적합의 징후를 보여 추가적인 튜닝이 필요함을 보였다.

---

# 4. U-NET vs. Mask-RCNN vs. DeepLabV3+

## 1. 모든 모델 학습 및 평가 후 Segmentation 예측 결과 비교

### (1) 학습 및 평가 실행

![{75148D8A-3A9C-4C58-9071-C5DAD1347950}.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/57512d83-727f-42a4-ac09-903c8850c224/75148D8A-3A9C-4C58-9071-C5DAD1347950.png)

⇒ 세가지 모델을 비교한 결과, 

Mask R-CNN >> DeepLabv3+ >> U-Net 순서로 세그멘테이션 감지가 잘 되는 것으로 보인다.

### (2) 각 모델 세그멘테이션 결과 시각화

- 이미지 예시 1

![모든모델비교1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/9cdd7990-ba26-499d-8d8f-96ef7155a36c/%EB%AA%A8%EB%93%A0%EB%AA%A8%EB%8D%B8%EB%B9%84%EA%B5%901.png)

- 이미지 예시 2

![모든모델비교2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/bddaf8c2-77ac-4f65-bb72-87cd9b2d0fcc/%EB%AA%A8%EB%93%A0%EB%AA%A8%EB%8D%B8%EB%B9%84%EA%B5%902.png)

- 이미지 예시 3

![모든모델비교3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8e93c4f1-6ad9-4a70-8e85-041046be0f87/a24f053f-8392-474a-b12c-91f84550bd9b/%EB%AA%A8%EB%93%A0%EB%AA%A8%EB%8D%B8%EB%B9%84%EA%B5%903.png)

## 시연영상
(DeepLabV3+)
![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/e88edaf2-1b40-47f6-a3fb-821d46000a67)

(U-Net)
![segmentation_imtovi_output-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/95a33910-a999-4a16-b24f-d36ecdd529d5)

(Mask R-CNN)
![Y2meta app-parkingspace_video_output_maskrcnn-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/9962bee8-c208-4364-8591-b293a0074c2e)

