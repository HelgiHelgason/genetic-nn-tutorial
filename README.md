# genetic-nn-tutorial
Basic tutorial for training feed-forward neural networks with genetic algorithms

This code uses a genetic algorithm to train a feed forward neural network to learn to approximate the cos(x) function. It is intentionally written to be easily understandable, at the cost of it's size.

It has no dependencies except Numpy (and could easily be modified to not even require that).

Note that this is written and published only with education purposes in mind, real-world training of neural networks is generally much more efficient with backpropagation.

Genetic algorithms are worth understanding because, while being resource intensive, they make no requirements for end-to-end differentiability, can easily handle discrete variables and are easy to code.

Typical output (will vary, the process is highly dependant on random intial conditions):

`Best one in generation 0 has error [[ 12.59287846]]

Best one in generation 1 has error [[ 2.78300017]]

Best one in generation 2 has error [[ 2.78300017]]

Best one in generation 3 has error [[ 2.78300017]]

Best one in generation 4 has error [[ 2.78300017]]

Best one in generation 5 has error [[ 2.78300017]]

Best one in generation 6 has error [[ 2.78300017]]

Best one in generation 7 has error [[ 2.78300017]]

Best one in generation 8 has error [[ 2.72173251]]

Best one in generation 9 has error [[ 2.72173251]]

Best one in generation 10 has error [[ 2.72173251]]

Best one in generation 11 has error [[ 2.09105546]]

Best one in generation 12 has error [[ 2.09105546]]

Best one in generation 13 has error [[ 2.09105546]]

Best one in generation 14 has error [[ 2.09105546]]

Best one in generation 15 has error [[ 2.09105546]]

Best one in generation 16 has error [[ 2.09105546]]

Best one in generation 17 has error [[ 2.09105546]]

Best one in generation 18 has error [[ 2.09105546]]

Best one in generation 19 has error [[ 2.07175697]]

Best one in generation 20 has error [[ 2.07175697]]

Best one in generation 21 has error [[ 2.07175697]]

Best one in generation 22 has error [[ 2.07175697]]

Best one in generation 23 has error [[ 2.07175697]]

Best one in generation 24 has error [[ 2.07175697]]

Best one in generation 25 has error [[ 2.07175697]]

Best one in generation 26 has error [[ 0.82709056]]

Best one in generation 27 has error [[ 0.82709056]]

Best one in generation 28 has error [[ 0.82709056]]

Best one in generation 29 has error [[ 0.82709056]]

Best one in generation 30 has error [[ 0.30601185]]

Best one in generation 31 has error [[ 0.30601185]]

Best one in generation 32 has error [[ 0.30601185]]

Best one in generation 33 has error [[ 0.30601185]]

Best one in generation 34 has error [[ 0.30601185]]

Best one in generation 35 has error [[ 0.30601185]]

Best one in generation 36 has error [[ 0.30601185]]

Best one in generation 37 has error [[ 0.30601185]]

Best one in generation 38 has error [[ 0.30601185]]

Best one in generation 39 has error [[ 0.27020429]]

Best one in generation 40 has error [[ 0.27020429]]

Best one in generation 41 has error [[ 0.27020429]]

Best one in generation 42 has error [[ 0.27020429]]

Best one in generation 43 has error [[ 0.18469612]]

Best one in generation 44 has error [[ 0.12397454]]

Best one in generation 45 has error [[ 0.12397454]]

Best one in generation 46 has error [[ 0.12397454]]

Best one in generation 47 has error [[ 0.12397454]]

Best one in generation 48 has error [[ 0.12397454]]

Best one in generation 49 has error [[ 0.12397454]]

Best one in generation 50 has error [[ 0.12397454]]

Best one in generation 51 has error [[ 0.12397454]]

Best one in generation 52 has error [[ 0.12397454]]

Best one in generation 53 has error [[ 0.12397454]]

Best one in generation 54 has error [[ 0.12397454]]

Best one in generation 55 has error [[ 0.12397454]]

Best one in generation 56 has error [[ 0.12397454]]

Best one in generation 57 has error [[ 0.12397454]]

Best one in generation 58 has error [[ 0.12397454]]

Best one in generation 59 has error [[ 0.12397454]]

Best one in generation 60 has error [[ 0.12397454]]

Best one in generation 61 has error [[ 0.12397454]]

Best one in generation 62 has error [[ 0.12397454]]

Best one in generation 63 has error [[ 0.12397454]]

Best one in generation 64 has error [[ 0.12397454]]

Best one in generation 65 has error [[ 0.12397454]]

Best one in generation 66 has error [[ 0.12397454]]

Best one in generation 67 has error [[ 0.12397454]]

Best one in generation 68 has error [[ 0.12397454]]

Best one in generation 69 has error [[ 0.12397454]]

Best one in generation 70 has error [[ 0.12397454]]

Best one in generation 71 has error [[ 0.12397454]]

Best one in generation 72 has error [[ 0.12397454]]

Best one in generation 73 has error [[ 0.12397454]]

Best one in generation 74 has error [[ 0.12397454]]

Best one in generation 75 has error [[ 0.12397454]]

Best one in generation 76 has error [[ 0.12397454]]

Best one in generation 77 has error [[ 0.12397454]]

Best one in generation 78 has error [[ 0.12397454]]

Best one in generation 79 has error [[ 0.12397454]]

Best one in generation 80 has error [[ 0.12397454]]

Best one in generation 81 has error [[ 0.12397454]]

Best one in generation 82 has error [[ 0.12397454]]

Best one in generation 83 has error [[ 0.12397454]]

Best one in generation 84 has error [[ 0.12397454]]

Best one in generation 85 has error [[ 0.1031022]]

Best one in generation 86 has error [[ 0.1031022]]

Best one in generation 87 has error [[ 0.1031022]]

Best one in generation 88 has error [[ 0.1031022]]

Best one in generation 89 has error [[ 0.1031022]]

Best one in generation 90 has error [[ 0.1031022]]

Best one in generation 91 has error [[ 0.1031022]]

Best one in generation 92 has error [[ 0.07956279]]

Best one in generation 93 has error [[ 0.07956279]]

Best one in generation 94 has error [[ 0.07956279]]

Best one in generation 95 has error [[ 0.07956279]]

Best one in generation 96 has error [[ 0.07956279]]

Best one in generation 97 has error [[ 0.07956279]]

Best one in generation 98 has error [[ 0.07956279]]

Best one in generation 99 has error [[ 0.07956279]]

From X= -0.04 we wanted  0.999200106661  and got  [[ 1.04551719]]

From X= -0.36 we wanted  0.935896823678  and got  [[ 0.91471741]]

From X= 0.53 we wanted  0.862807070515  and got  [[ 0.85238186]]

From X= -0.4 we wanted  0.921060994003  and got  [[ 0.89261504]]

From X= 0.39 we wanted  0.924909059857  and got  [[ 0.92753762]]

From X= 0.24 we wanted  0.971337974852  and got  [[ 0.97965819]]

From X= 0.29 we wanted  0.958243875513  and got  [[ 0.96638876]]

From X= -0.93 we wanted  0.597833982287  and got  [[ 0.61892997]]

From X= -0.08 we wanted  0.996801706303  and got  [[ 1.04170355]]

From X= 0.8 we wanted  0.696706709347  and got  [[ 0.67751945]]

From X= 0.41 we wanted  0.917120822817  and got  [[ 0.91680108]]

From X= 0.01 we wanted  0.999950000417  and got  [[ 1.04126602]]

From X= -0.94 we wanted  0.589788025031  and got  [[ 0.61368378]]

From X= 0.4 we wanted  0.921060994003  and got  [[ 0.92216935]]

From X= 0.51 we wanted  0.872744507646  and got  [[ 0.86311839]]

From X= -0.37 we wanted  0.932327345606  and got  [[ 0.90919182]]

From X= -0.17 we wanted  0.98558476691  and got  [[ 1.01394002]]

From X= -0.82 we wanted  0.682221207288  and got  [[ 0.67663809]]

From X= 0.06 we wanted  0.998200539935  and got  [[ 1.02742811]]

From X= 0.22 we wanted  0.975897449331  and got  [[ 0.98496596]]

From X= 0.94 we wanted  0.589788025031  and got  [[ 0.56510386]]

From X= 0.16 we wanted  0.987227283376  and got  [[ 1.00088926]]

From X= -0.19 we wanted  0.982004235117  and got  [[ 1.00617646]]

From X= 0.35 we wanted  0.939372712847  and got  [[ 0.94901069]]

From X= -0.63 we wanted  0.808027508312  and got  [[ 0.78306065]]

From X= 0.05 we wanted  0.998750260395  and got  [[ 1.03019056]]

From X= -0.9 we wanted  0.621609968271  and got  [[ 0.63466855]]

From X= -0.99 we wanted  0.548689860582  and got  [[ 0.58745281]]

From X= -0.73 we wanted  0.745174402345  and got  [[ 0.73255576]]

From X= -0.02 we wanted  0.999800006667  and got  [[ 1.04675805]]

From X= 0.78 we wanted  0.710913538012  and got  [[ 0.69063216]]

From X= 0.3 we wanted  0.955336489126  and got  [[ 0.96373488]]

From X= -0.24 we wanted  0.971337974852  and got  [[ 0.98102451]]

From X= -0.53 we wanted  0.862807070515  and got  [[ 0.82788038]]

From X= -0.69 we wanted  0.771246014997  and got  [[ 0.75641887]]

From X= 0.38 we wanted  0.928664635577  and got  [[ 0.93290589]]

From X= -0.16 we wanted  0.987227283376  and got  [[ 1.0178218]]

From X= -0.95 we wanted  0.581683089464  and got  [[ 0.60843758]]

From X= 0.93 we wanted  0.597833982287  and got  [[ 0.57423955]]

From X= -0.51 we wanted  0.872744507646  and got  [[ 0.83684432]]

From X= -0.09 we wanted  0.995952733012  and got  [[ 1.04075014]]

From X= -0.61 we wanted  0.819648017845  and got  [[ 0.79202459]]

From X= -0.98 we wanted  0.557022546766  and got  [[ 0.59269901]]

From X= 0.5 we wanted  0.87758256189  and got  [[ 0.86848666]]

From X= 0.69 we wanted  0.771246014997  and got  [[ 0.74588676]]

From X= 0.75 we wanted  0.731688868874  and got  [[ 0.70945492]]

From X= -0.28 we wanted  0.961055438311  and got  [[ 0.95892214]]

From X= -0.81 we wanted  0.689498432952  and got  [[ 0.68249237]]

From X= -0.91 we wanted  0.613745749489  and got  [[ 0.62942235]]

From X= 0.15 we wanted  0.988771077936  and got  [[ 1.00354315]]

From X= -0.7 we wanted  0.764842187284  and got  [[ 0.75132953]]

From X= 0.25 we wanted  0.968912421711  and got  [[ 0.9770043]]

From X= 0.37 we wanted  0.932327345606  and got  [[ 0.93827416]]

From X= 0.14 we wanted  0.990215996213  and got  [[ 1.00619703]]

From X= 0.74 we wanted  0.73846855873  and got  [[ 0.71552689]]

From X= 0.48 we wanted  0.886994922779  and got  [[ 0.8792232]]

From X= 0.52 we wanted  0.867819179678  and got  [[ 0.85775012]]

From X= -0.1 we wanted  0.995004165278  and got  [[ 1.03979673]]

From X= -0.34 we wanted  0.942754665528  and got  [[ 0.92576859]]

From X= -0.76 we wanted  0.724836010741  and got  [[ 0.71378198]]

From X= -0.59 we wanted  0.8309406791  and got  [[ 0.80098854]]

From X= 0.07 we wanted  0.997551000253  and got  [[ 1.02477422]]

From X= -0.85 we wanted  0.659983145885  and got  [[ 0.66089951]]

From X= -0.03 we wanted  0.999550033749  and got  [[ 1.0464706]]

From X= -0.42 we wanted  0.913088940312  and got  [[ 0.88156386]]

From X= 0.19 we wanted  0.982004235117  and got  [[ 0.99292761]]

From X= 0.04 we wanted  0.999200106661  and got  [[ 1.03295942]]

From X= -0.07 we wanted  0.997551000253  and got  [[ 1.04265696]]

From X= -0.66 we wanted  0.789992231497  and got  [[ 0.76961473]]

From X= -1.0 we wanted  0.540302305868  and got  [[ 0.58220662]]

From X= 0.86 we wanted  0.652437468164  and got  [[ 0.63818939]]

From X= -0.96 we wanted  0.573519986072  and got  [[ 0.60319139]]

From X= 0.46 we wanted  0.896052497526  and got  [[ 0.88995974]]

From X= -0.78 we wanted  0.710913538012  and got  [[ 0.70126614]]

From X= 0.42 we wanted  0.913088940312  and got  [[ 0.91143281]]

From X= 0.7 we wanted  0.764842187284  and got  [[ 0.73981479]]

From X= -0.38 we wanted  0.928664635577  and got  [[ 0.90366622]]

From X= -0.71 we wanted  0.758361875991  and got  [[ 0.7450716]]

From X= 0.98 we wanted  0.557022546766  and got  [[ 0.52856109]]

From X= 0.49 we wanted  0.88233285861  and got  [[ 0.87385493]]

From X= 0.82 we wanted  0.682221207288  and got  [[ 0.66408245]]

From X= 0.99 we wanted  0.548689860582  and got  [[ 0.5194254]]

From X= -0.56 we wanted  0.847255111013  and got  [[ 0.81443446]]

From X= -0.67 we wanted  0.783821665881  and got  [[ 0.76513276]]

From X= -0.74 we wanted  0.73846855873  and got  [[ 0.72629783]]

From X= 0.83 we wanted  0.674875760071  and got  [[ 0.65785241]]

From X= -0.55 we wanted  0.85252452206  and got  [[ 0.81891643]]

From X= -0.92 we wanted  0.605820156643  and got  [[ 0.62417616]]

From X= -0.23 we wanted  0.973666395005  and got  [[ 0.9865501]]

From X= -0.47 we wanted  0.891568288195  and got  [[ 0.85477221]]

From X= -0.18 we wanted  0.983843692788  and got  [[ 1.01005824]]

From X= -0.86 we wanted  0.652437468164  and got  [[ 0.65565332]]

From X= 0.84 we wanted  0.667462825841  and got  [[ 0.65162237]]

From X= -0.11 we wanted  0.993956097957  and got  [[ 1.03723069]]

From X= -0.44 we wanted  0.90475166322  and got  [[ 0.87051267]]

From X= -0.01 we wanted  0.999950000417  and got  [[ 1.04571423]]

From X= 0.9 we wanted  0.621609968271  and got  [[ 0.60164663]]

From X= -0.2 we wanted  0.980066577841  and got  [[ 1.00229468]]

From X= 0.47 we wanted  0.891568288195  and got  [[ 0.88459147]]

From X= 0.58 we wanted  0.836462649915  and got  [[ 0.8191097]]

From X= 0.26 we wanted  0.966389978135  and got  [[ 0.97435042]]

From X= 0.27 we wanted  0.963770896366  and got  [[ 0.97169653]]

From X= 0.88 we wanted  0.637151144199  and got  [[ 0.61991801]]

From X= 0.77 we wanted  0.717910669611  and got  [[ 0.69718851]]

From X= 0.44 we wanted  0.90475166322  and got  [[ 0.90069627]]

From X= -0.88 we wanted  0.637151144199  and got  [[ 0.64516093]]

From X= 0.59 we wanted  0.8309406791  and got  [[ 0.81221108]]

From X= -0.41 we wanted  0.917120822817  and got  [[ 0.88708945]]

From X= -0.3 we wanted  0.955336489126  and got  [[ 0.94787096]]

From X= 0.89 we wanted  0.629412026574  and got  [[ 0.61078232]]

From X= -0.45 we wanted  0.900447102353  and got  [[ 0.86498708]]

From X= -0.75 we wanted  0.731688868874  and got  [[ 0.72003991]]

From X= -0.65 we wanted  0.796083798549  and got  [[ 0.7740967]]

From X= -0.54 we wanted  0.857708681364  and got  [[ 0.8233984]]

From X= 0.31 we wanted  0.952333569886  and got  [[ 0.961081]]

From X= 0.11 we wanted  0.993956097957  and got  [[ 1.01415869]]

From X= 0.96 we wanted  0.573519986072  and got  [[ 0.54683248]]

From X= -0.35 we wanted  0.939372712847  and got  [[ 0.920243]]

From X= 0.62 we wanted  0.813878456663  and got  [[ 0.79151523]]

From X= -0.68 we wanted  0.777572718751  and got  [[ 0.76069849]]

From X= -0.43 we wanted  0.908965749675  and got  [[ 0.87603827]]

From X= 0.72 we wanted  0.751805729141  and got  [[ 0.72767084]]

From X= -0.32 we wanted  0.949235418082  and got  [[ 0.93681977]]

From X= 0.85 we wanted  0.659983145885  and got  [[ 0.64539233]]

From X= 0.45 we wanted  0.900447102353  and got  [[ 0.89532801]]

From X= 0.64 we wanted  0.802095757884  and got  [[ 0.77771799]]

From X= 0.71 we wanted  0.758361875991  and got  [[ 0.73374281]]

From X= -0.84 we wanted  0.667462825841  and got  [[ 0.6661457]]

From X= -0.57 we wanted  0.841900975162  and got  [[ 0.80995248]]

From X= 0.54 we wanted  0.857708681364  and got  [[ 0.84670417]]

From X= -0.33 we wanted  0.946042343528  and got  [[ 0.93129418]]

From X= -0.49 we wanted  0.88233285861  and got  [[ 0.84580827]]

From X= 0.66 we wanted  0.789992231497  and got  [[ 0.76410268]]

From X= 0.2 we wanted  0.980066577841  and got  [[ 0.99027373]]

From X= 0.18 we wanted  0.983843692788  and got  [[ 0.99558149]]

From X= 0.55 we wanted  0.85252452206  and got  [[ 0.83980555]]

From X= 0.56 we wanted  0.847255111013  and got  [[ 0.83290694]]

From X= -0.31 we wanted  0.952333569886  and got  [[ 0.94234537]]

From X= 0.81 we wanted  0.689498432952  and got  [[ 0.67031249]]

From X= 0.23 we wanted  0.973666395005  and got  [[ 0.98231207]]

From X= 0.08 we wanted  0.996801706303  and got  [[ 1.02212034]]

From X= 0.43 we wanted  0.908965749675  and got  [[ 0.90606454]]

From X= 0.68 we wanted  0.777572718751  and got  [[ 0.75195873]]

From X= -0.26 we wanted  0.966389978135  and got  [[ 0.96997332]]

From X= -0.13 we wanted  0.991561893715  and got  [[ 1.02946714]]

From X= -0.27 we wanted  0.963770896366  and got  [[ 0.96444773]]

From X= 0.28 we wanted  0.961055438311  and got  [[ 0.96904265]]

From X= 0.13 we wanted  0.991561893715  and got  [[ 1.00885092]]

From X= 0.32 we wanted  0.949235418082  and got  [[ 0.95842711]]

From X= 0.17 we wanted  0.98558476691  and got  [[ 0.99823538]]`