#DocGarbanzo Car

## Donkey Car

Donkey&reg; Car is an open-source project for building self-driving RC cars. 
It is highly hackable and a platform for experimentation. Donkey cars are:

* Fast
* Cheap
* Out of control

The Donkey Car code and documentation is available here:

* [GitHub](https://github.com/autorope/donkeycar)
* [Documentation](https://docs.donkeycar.com)


``` mermaid
graph TD;
    A[Donkey Version]-->B[<= 4.4.X];
    A[Donkey Version]-->C[>= 4.4.X, i.e. main]
    B-->D[<b>Jetson Nano</b> <br> Jetpack 4.5.2 <br> Python 3.6 <br> Tensorflow 2.3.1];
    B-->E[<b>Jetson Xavier</b> <br> Not supported];
    C-->F[<b>Jetson Nano</b> <br> Jetpack 4.6.2 <br> Python 3.9 <br> Tensorflow 2.9]
    C-->G[<b>Jetson Xavier</b> <br> Jetpack 5.0.2 <br> Python 3.8 <br> Tensorflow 2.9]
```


## Car Build

[Here](buggy.md) is a build log of the bespoke Donkey Car that I built on an 
alternative RC car platform. 


![DocGarbanzoCar](./assets/front.jpeg)




