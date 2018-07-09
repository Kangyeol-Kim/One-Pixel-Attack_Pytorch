TERMS
* black-box attack - information available is the probability labels
* ground-true - notion of data that is "known" to be correct.
* meta-heuristic - heuristic designed to find, generate, or select a heuristic (partial search algorithm) that may *provide a sufficiently good solution to an optimization problem*, especially with incomplete or imperfect information or limited computation capacity
* DE - https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
*Increasing the probability label values of the target classes*




cutting the input space using very low-dimensional slices..?

number of pixels = units

most data points in the input space are gathered near to the boundaries

The one-pixel modification can be seen as perturbing the data point along a direction parallel to the axis of one of the n dimensions

* Differential evolution(DE) for generating adversarial images
  * meta-heuristic, global Optima
  * Require Less Information from Target System - Only need probability of label. NO NEED TO KNOW GRADIENT

### Setting

400 init point

100 iter maximum


                            CIFAR-10                                      ImageNet
Early-stop criteria: probability of target class exceed 50 %     True class label below 5%

initialized pop    :        U(1, 32)                                     U(1, 227)

RGB value : N(128, 127)


fitness function : probailistic label of the target class

### Evaluation
                          non-target attack                           target attack
Success Rate -           classified by the other class           probability of target class

Adversarial Probability Labels(confidence)                          target class의 확률 모아서 평균







vicinity - 부근
curvature - 곡률
intuition of convolution : colah bolg
