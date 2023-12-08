# Final project - AY 2023/24

* [Project - FAQ](https://virtuale.unibo.it/mod/page/view.php?id=1405067)
* [Slides](https://docs.google.com/presentation/d/1TTN1H3GdnaswGXW63SuSvD4CsI7HB9lkYuwXRMQp2ks/edit#slide=id.p)
* [Assignment: SemEval 2024 Task 10, SUBTASK iii](https://lcs2.in/SemEval2024-EDiReF/)
* [Dataset](https://drive.google.com/drive/folders/1YgUU9nwFr9UiJKmGbFS9ByuL5fQWp8MO)

TODO

* Stampare ROC Curve per capire la threshold migliore per il binary classifier
* usare il metodo evaluate al posto dei due blocchi di codice per i modelli freezed e fully finetuned
* applicare la threshold a test_pred_tri(\_full/\_freezed) creando un nuovo array, così si possono testare threshold diverse senza fare un'altra run 
* Error analysis
* Valutare input utternce separato dal contesto
* 5 seed diversi


IDEAS

* undersample (undersample) the majority (minority) class
* data agumentation for the minority class samples - Implemented a little bit, needs more tests
* bigger update (loss) for the trigger errors - Implemented, needs tests
* separate training of the two classification heads