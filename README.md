# genetic-nn-tutorial
Basic tutorial for training feed-forward neural networks with genetic algorithms

This code uses a genetic algorithm to train a feed forward neural network to learn to approximate the cosine function.

The only dependencies are Numpy and Matplotlib.

Note that this is written and published only with education purposes in mind, real-world training of neural networks is generally much more efficient with backpropagation.

Genetic algorithms are worth understanding because, while being time-consuming, they make no requirements for end-to-end differentiability and are easy to code.

## Typical output
(Will vary, the process is highly dependant on random initial conditions):

```
Starting evolution
Best solution in generation 0 has error 7.61236139532
Best solution in generation 1 has error 7.61236139532
Best solution in generation 2 has error 7.61236139532
Best solution in generation 3 has error 3.27220677579
Best solution in generation 4 has error 3.27220677579
Best solution in generation 5 has error 3.1588575889
Best solution in generation 6 has error 3.1588575889
Best solution in generation 7 has error 2.89068589998
Best solution in generation 8 has error 2.83810113324
Best solution in generation 9 has error 2.83810113324
Best solution in generation 10 has error 2.83810113324
.........
Best solution in generation 90 has error 0.0486103407126
Best solution in generation 91 has error 0.0480764954396
Best solution in generation 92 has error 0.0480764954396
Best solution in generation 93 has error 0.0480764954396
Best solution in generation 94 has error 0.0480764954396
Best solution in generation 95 has error 0.0480764954396
Best solution in generation 96 has error 0.0456230812112
Best solution in generation 97 has error 0.0456230812112
Best solution in generation 98 has error 0.0456230812112
Best solution in generation 99 has error 0.0456230812112
Evolution done
Starting test phase
From X= -0.76 we wanted 0.724836010741 and got 0.712605095881
From X= -0.15 we wanted 0.988771077936 and got 1.00492015883
From X= -0.37 we wanted 0.932327345606 and got 0.914941461067
From X= -0.47 we wanted 0.891568288195 and got 0.864431013776
From X= 0.33 we wanted 0.946042343528 and got 0.963742326553
.........
From X= 0.32 we wanted 0.949235418082 and got 0.965453716445
From X= 0.48 we wanted 0.886994922779 and got 0.88210409188
From X= -0.77 we wanted 0.717910669611 and got 0.707369719402
From X= 0.55 we wanted 0.85252452206 and got 0.840726083929
From X= -0.79 we wanted 0.703845315652 and got 0.696898966444

```


Copyright (c) 2016 Helgi Páll Helgason.

