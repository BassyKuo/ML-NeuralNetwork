# ML-NeuralNetwork
Implementation of a feedforward neural network learning algorithm by the stochastic gradient descent (SGD) algorithm and Backpropragation.

## Execution Version
Python 3.5.2

## Requirements
* Numpy >= 1.12.1

## Usage
```
$ python3 main.py -h
```

For example,
```
$ python3 main.py train --epoch 10000 --lr 0.01 --lam 0 --depth 3 > out.txt
```

### Usage of Training Phase
```
$ python3 main.py train

[Thread-0][epoch 0]	Val Loss: 343.0185621703511
[Thread-2][epoch 0]	Val Loss: 339.0184810020424
[Thread-3][epoch 0]	Val Loss: 345.82376663371275
[Thread-1][epoch 0]	Val Loss: 325.75220661483223
[Thread-4][epoch 0]	Val Loss: 340.74480979618494
[Thread-0][epoch 1]	Val Loss: 339.25992781363317
[Thread-2][epoch 1]	Val Loss: 334.6588013083471
[Thread-3][epoch 1]	Val Loss: 340.17716599320755
[Thread-1][epoch 1]	Val Loss: 317.67991183594546
[Thread-0][epoch 2]	Val Loss: 331.4401588506384
...(etc.)

[OPTIMAL]
+ Val Loss: 25.083101961174567
+ Depth: 3
+ Lambda: 0.7
Test Error: 29.462166641014022

>> Record `depth`, `sizes`, `weight_{N}` for each layer in SGD_hypothesis_header.csv

>> Record `depth`, `lambda`, `val_loss`, `test_loss`, `epoch`, 'sizes' for each thread in log/table-10000ep-06192336.txt
```
> __[ Train multi-layers or regularization parameter simultaneously ]:__
> ```python
> #./main.py
> depth_trace = 1	# the number of different depth sizes for training (between 1 ~ 5)
> lam_trace	= 1		# the number of different lambda for training (between 1 ~ 5)
> ```


### Usage of Testing Phase
```
$ python3 main.py test

Load the hypothesis parameters file [SGD_hypothesis_header.csv]: 
Load the testing file [energy_efficiency_cooling_load_testing.csv]: 
[Performance]
Test Error: 2.714527031447499
```


## Data Format
```
<label>,<feature_1>,<feature_2>, ... ,<feature_n>
```

For example,
```
13.99,-0.9941578053736886,0.9940434404178762,0.4264020500151475,0.9918732160897312,-0.9900926215671461,0.33086559963852746,-0.496691523780351,0.9938344116583973
33.87,-0.05523098918742686,-0.16567390673631266,0.14213401667171577,-0.3306244053632438,0.9900926215671462,0.33086559963852746,-0.496691523780351,-0.1987668823316795
32.77,0.5523098918742717,-0.6626956269452507,-0.142134016671716,-0.7714569458475686,0.9900926215671462,0.33086559963852746,0.24834576189017543,-0.1987668823316795
16.44,-0.6075408810616988,0.497021720208938,-0.42640205001514775,0.9918732160897312,-0.9900926215671461,-0.9925967989155825,0.9933830475607017,-0.1987668823316795
33.88,-0.22092395674970844,0.0,0.994938116702011,-0.7714569458475686,0.9900926215671462,0.9925967989155824,-0.496691523780351,0.9938344116583973
15.37,-0.6075408810616988,0.497021720208938,-0.42640205001514775,0.9918732160897312,-0.9900926215671461,-0.9925967989155825,0.24834576189017543,0.5963006469950383
```
