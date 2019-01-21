# Automatic transcription using embedding latent space

This work adresses the problem of automatic transcription of musical audio pieces to midi sheets. We propose a method
based on recent research by Dorfer et al.[4] that learns joint embedding spaces for short excerpts of musical audio and
their respective counterparts in midi scores, using multimodal convolutional neural networks. The dataset is based on Pianomidi.de, a midi sheet collection of classical music played on piano and splitted by Poliner and Ellis [2006] comprising 314
midi scores from 25 composers.

## Report

The report associated with this project can be found [here]

## How to use

### Dependecies

What things you need to install the software and how to install them

```shell
pip3 install numpy
pip3 install torch torchvision
pip3 install pypianoroll
pip3 install midi2audio
pip3 install pyFluidSynth
pip3 install librosa


```

## Authors

* **Guilhem Marion** 
* **Valérian Fraisse** 
* **Clément Le Moine Veillon**
* **Félix Rohrlich**
* **Gabriel Dias Neto**


## License

This project is Open Source.

## Acknowledgments

* We would like to thank Mathieu Prang, PhD student at IRCAM and Philippe Esling, associate professor at IRCAM and Sorbonne Université, for their help and support during this project.