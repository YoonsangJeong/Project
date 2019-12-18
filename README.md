# Project

1. 모델 설정 (class DNN)
- Sequential 클라스를 사용하여 DNN 모델을 설정 합니다.
- DNN.py에서 설정된 모델은 2개의 Hidden Layer와 1개의 softmax측으로 이루어져 있는 DNN 모델입니다.
- 활성화함수로 Relu를 사용하며, 각 Hidden Layer에는 Dropout 기법을 사용하였습니다.
- compile()함수를 사용하여 손실함수 및 최적화 방법을 정의합니다.

2. 데이터 셋 설정 (def Data_func)
-사용된 데이터 셋은 10개의 클래스를 가지는 cifar-10 이미지 데이터입니다.
-각각의 이미지는 32X32 RGB 컬러 이미지이며, 트레이닝에 50000개, 테스트에 10000개의 이미지가 각각 사용되었습니다.
-각 이미지는 1X3072 형태의 feature vecture로 변환되어 DNN모델에 입력됩니다.
-각 feature는 0~1 값이 되도록 정규화됩니다.

3. 학습과정 Plot (def plot_acc, def plot_acc)
-각 epoch에서 DNN모델의 loss와 accuracy를 그래프로 도시하는 함수입니다.

4. 모델 학습 및 평가 (def main)
-설정된 DNN 모델 구조와 데이터 셋을 사용하여 DNN 모델을 학습 및 평가합니다.
-fit()함수와 트레이닝 데이터 셋을 통해 설정된 DNN 모델을 학습시킵니다. 이때, Batch size는 100, epoch은 150회입니다.
-합습된 모델은 evaluate()함수를 사용하여 테스트 데이터 셋을 사용하여 평가됩니다. 마찬가지로 Batch size는 100입니다.

-Result
최종 Loss와 Accuracy는 각각 1.5588648295402527, 0.44600000977516174이 출력됩니다.




Reference
[1] Github,"KERASPP" , https://github.com/RossSong/keraspp (2019.12.17)

[2] Github.io,"케라스 이야기",  https://tykimos.github.io/2017/01/27/Keras_Talk/ (2019.12.17)
